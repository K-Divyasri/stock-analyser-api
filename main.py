"""
Stock Returns Analyser — FastAPI Backend
Run locally:  uvicorn main:app --reload
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings("ignore")

app = FastAPI(title="Stock Returns Analyser API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

FEATURE_COLS = [
    "ret_lag1","ret_lag2","ret_lag3","ret_lag5",
    "roll_mean_5","roll_mean_10","roll_mean_20",
    "roll_std_5","roll_std_10","roll_std_20",
    "momentum_5","momentum_10","momentum_20",
    "rsi_14","vol_ratio",
]

# ── Feature engineering ───────────────────────────────────────────────────────
def make_features(prices: pd.Series) -> pd.DataFrame:
    df      = pd.DataFrame(index=prices.index)
    log_ret = np.log(prices / prices.shift(1))
    for lag in [1,2,3,5]:
        df[f"ret_lag{lag}"] = log_ret.shift(lag)
    for w in [5,10,20]:
        df[f"roll_mean_{w}"] = log_ret.rolling(w).mean()
        df[f"roll_std_{w}"]  = log_ret.rolling(w).std()
        df[f"momentum_{w}"]  = prices / prices.shift(w) - 1
    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi_14"]    = 100 - (100 / (1 + gain / (loss + 1e-9)))
    df["vol_ratio"] = log_ret.rolling(5).std() / (log_ret.rolling(20).std() + 1e-9)
    df["target"]    = (log_ret.shift(-1) > 0).astype(int)
    return df.dropna()

# ── In-memory model cache ─────────────────────────────────────────────────────
_model_cache: dict = {}

def get_model(ticker: str):
    if ticker in _model_cache:
        return _model_cache[ticker]
    prices = yf.download(ticker, period="5y",
                         auto_adjust=True, progress=False)["Close"].squeeze()
    if prices.empty:
        raise HTTPException(404, f"No data for {ticker}")
    df     = make_features(prices)
    X, y   = df[FEATURE_COLS], df["target"]
    split  = int(len(X) * 0.8)
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X.iloc[:split])
    X_te   = scaler.transform(X.iloc[split:])
    clf    = RandomForestClassifier(n_estimators=200, max_depth=6,
                                    min_samples_leaf=20, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y.iloc[:split])
    acc    = clf.score(X_te, y.iloc[split:])
    _model_cache[ticker] = (clf, scaler, acc)
    return clf, scaler, acc

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Stock Returns Analyser API — see /docs for endpoints"}

@app.get("/stats/{ticker}")
def get_stats(ticker: str):
    """Return summary statistics for a ticker."""
    ticker = ticker.upper()
    prices  = yf.download(ticker, period="5y",
                          auto_adjust=True, progress=False)["Close"].squeeze()
    if prices.empty:
        raise HTTPException(404, f"No data for {ticker}")
    log_ret = np.log(prices / prices.shift(1)).dropna()
    return {
        "ticker":           ticker,
        "mean_daily":       round(float(log_ret.mean()), 6),
        "std_daily":        round(float(log_ret.std()),  6),
        "skewness":         round(float(log_ret.skew()), 4),
        "kurtosis":         round(float(log_ret.kurt()), 4),
        "ann_return":       round(float(log_ret.mean() * 252), 4),
        "ann_volatility":   round(float(log_ret.std()  * np.sqrt(252)), 4),
        "sharpe":           round(float(log_ret.mean() * 252 /
                                        (log_ret.std() * np.sqrt(252))), 4),
        "trading_days":     len(log_ret),
    }

@app.get("/correlation")
def get_correlation(tickers: str = "AAPL,MSFT,GOOG,AMZN,TSLA"):
    """Return correlation matrix for a comma-separated list of tickers."""
    ticker_list = [t.strip().upper() for t in tickers.split(",")]
    raw   = yf.download(ticker_list, period="5y",
                        auto_adjust=True, progress=False)
    close = raw["Close"].dropna()
    log_ret = np.log(close / close.shift(1)).dropna()
    corr = log_ret.corr().round(4)
    return corr.to_dict()

@app.get("/predict/{ticker}")
def predict(ticker: str):
    """Predict next-day direction for a ticker using a Random Forest."""
    ticker = ticker.upper()
    clf, scaler, acc = get_model(ticker)

    live   = yf.download(ticker, period="6mo",
                         auto_adjust=True, progress=False)["Close"].squeeze()
    feat   = make_features(live)
    if feat.empty:
        raise HTTPException(422, "Not enough data to compute features")

    X_live = scaler.transform(feat[FEATURE_COLS].iloc[[-1]])
    pred   = int(clf.predict(X_live)[0])
    prob   = clf.predict_proba(X_live)[0].tolist()

    return {
        "ticker":         ticker,
        "prediction":     "UP" if pred == 1 else "DOWN",
        "prob_up":        round(prob[1], 4),
        "prob_down":      round(prob[0], 4),
        "confidence":     round(max(prob), 4),
        "model_accuracy": round(acc, 4),
    }

@app.get("/returns/{ticker}")
def get_returns(ticker: str, days: int = 30):
    """Return the last N daily log returns for charting."""
    ticker  = ticker.upper()
    prices  = yf.download(ticker, period="1y",
                          auto_adjust=True, progress=False)["Close"].squeeze()
    log_ret = np.log(prices / prices.shift(1)).dropna().tail(days)
    return {
        "ticker": ticker,
        "dates":  log_ret.index.strftime("%Y-%m-%d").tolist(),
        "values": log_ret.round(6).tolist(),
    }
