"""
Stock Returns Analyser — FastAPI Backend (Fixed)
- Lighter Random Forest (fits in Render free 512MB RAM)
- Better error handling on every endpoint
- yfinance retry logic
- Pre-warms cache on startup for AAPL
Run locally:  uvicorn main:app --reload
"""
import warnings
warnings.filterwarnings("ignore")

import time
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="Stock Returns Analyser API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag5",
    "roll_mean_5", "roll_mean_10", "roll_mean_20",
    "roll_std_5",  "roll_std_10",  "roll_std_20",
    "momentum_5",  "momentum_10",  "momentum_20",
    "rsi_14",      "vol_ratio",
]

# ── yfinance helper with retry ────────────────────────────────────────────────
def download_with_retry(ticker: str, period: str, retries: int = 3) -> pd.Series:
    """Download price data with retries to handle yfinance flakiness."""
    for attempt in range(retries):
        try:
            raw = yf.download(
                ticker,
                period=period,
                auto_adjust=True,
                progress=False,
                timeout=30,
            )
            if raw.empty:
                raise ValueError("Empty response from yfinance")
            close = raw["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.squeeze()
            return close.dropna()
        except Exception as e:
            log.warning(f"Attempt {attempt+1}/{retries} failed for {ticker}: {e}")
            if attempt < retries - 1:
                time.sleep(2)
    raise HTTPException(
        status_code=503,
        detail=f"Could not download data for {ticker} after {retries} attempts. "
               "Yahoo Finance may be temporarily unavailable — try again in 30 seconds."
    )

# ── Feature engineering ───────────────────────────────────────────────────────
def make_features(prices: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame(index=prices.index)
    log_ret = np.log(prices / prices.shift(1))

    for lag in [1, 2, 3, 5]:
        df[f"ret_lag{lag}"] = log_ret.shift(lag)
    for w in [5, 10, 20]:
        df[f"roll_mean_{w}"] = log_ret.rolling(w).mean()
        df[f"roll_std_{w}"]  = log_ret.rolling(w).std()
        df[f"momentum_{w}"]  = prices / prices.shift(w) - 1

    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi_14"]    = 100 - (100 / (1 + gain / (loss + 1e-9)))
    df["vol_ratio"] = (
        log_ret.rolling(5).std() / (log_ret.rolling(20).std() + 1e-9)
    )
    df["target"] = (log_ret.shift(-1) > 0).astype(int)
    return df.dropna()

# ── Model cache ───────────────────────────────────────────────────────────────
_model_cache: dict = {}

def get_model(ticker: str):
    """Train and cache a Random Forest for the given ticker."""
    ticker = ticker.upper()
    if ticker in _model_cache:
        log.info(f"Cache hit for {ticker}")
        return _model_cache[ticker]

    log.info(f"Training model for {ticker}...")
    prices = download_with_retry(ticker, period="5y")

    if len(prices) < 100:
        raise HTTPException(
            status_code=422,
            detail=f"Not enough price history for {ticker} "
                   f"(got {len(prices)} days, need 100+)."
        )

    df    = make_features(prices)
    X, y  = df[FEATURE_COLS], df["target"]
    split = int(len(X) * 0.8)

    if split < 50:
        raise HTTPException(
            status_code=422,
            detail=f"Not enough training rows for {ticker} after feature engineering."
        )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X.iloc[:split])
    X_test  = scaler.transform(X.iloc[split:])

    # Lighter model — fits comfortably in 512 MB RAM on free Render tier
    clf = RandomForestClassifier(
        n_estimators=100,    # was 200 — halved to save memory
        max_depth=4,         # was 6   — shallower trees
        min_samples_leaf=30,
        max_features="sqrt",
        random_state=42,
        n_jobs=1,            # single thread is safer on free tier
    )
    clf.fit(X_train, y.iloc[:split])
    acc = float(clf.score(X_test, y.iloc[split:]))

    _model_cache[ticker] = (clf, scaler, acc)
    log.info(f"Model trained for {ticker} — accuracy {acc:.1%}")
    return clf, scaler, acc

# ── Startup: pre-warm AAPL so first real request is fast ─────────────────────
@app.on_event("startup")
async def warmup():
    try:
        log.info("Pre-warming model cache for AAPL...")
        get_model("AAPL")
        log.info("Warm-up complete.")
    except Exception as e:
        log.warning(f"Warm-up failed (non-fatal): {e}")

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Stock Returns Analyser API v1.1",
        "endpoints": [
            "/stats/{ticker}",
            "/predict/{ticker}",
            "/returns/{ticker}",
            "/correlation",
            "/docs",
        ],
    }

@app.get("/health")
def health():
    return {"status": "ok", "cached_tickers": list(_model_cache.keys())}

@app.get("/stats/{ticker}")
def get_stats(ticker: str):
    """Return annualised statistics for a single ticker."""
    ticker  = ticker.upper()
    prices  = download_with_retry(ticker, period="5y")
    log_ret = np.log(prices / prices.shift(1)).dropna()

    if len(log_ret) < 20:
        raise HTTPException(422, f"Not enough return data for {ticker}.")

    ann_ret = float(log_ret.mean() * 252)
    ann_vol = float(log_ret.std()  * np.sqrt(252))

    return {
        "ticker":         ticker,
        "trading_days":   int(len(log_ret)),
        "mean_daily":     round(float(log_ret.mean()), 6),
        "std_daily":      round(float(log_ret.std()),  6),
        "skewness":       round(float(log_ret.skew()), 4),
        "kurtosis":       round(float(log_ret.kurt()), 4),
        "ann_return":     round(ann_ret, 4),
        "ann_volatility": round(ann_vol, 4),
        "sharpe":         round(ann_ret / ann_vol, 4) if ann_vol != 0 else None,
    }

@app.get("/predict/{ticker}")
def predict(ticker: str):
    """Predict next-day direction using a cached Random Forest."""
    ticker       = ticker.upper()
    clf, scaler, acc = get_model(ticker)

    live = download_with_retry(ticker, period="6mo")
    feat = make_features(live)

    if feat.empty or len(feat) < 1:
        raise HTTPException(422, "Not enough recent data to compute features.")

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
def get_returns(ticker: str, days: int = 60):
    """Return the last N daily log returns (max 252)."""
    ticker  = ticker.upper()
    days    = min(days, 252)
    prices  = download_with_retry(ticker, period="1y")
    log_ret = np.log(prices / prices.shift(1)).dropna().tail(days)

    if log_ret.empty:
        raise HTTPException(422, f"Could not compute returns for {ticker}.")

    return {
        "ticker": ticker,
        "dates":  log_ret.index.strftime("%Y-%m-%d").tolist(),
        "values": [round(float(v), 6) for v in log_ret.tolist()],
    }

@app.get("/correlation")
def get_correlation(tickers: str = "AAPL,MSFT,GOOG,AMZN,TSLA"):
    """Return correlation matrix for comma-separated tickers."""
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]

    if len(ticker_list) < 2:
        raise HTTPException(422, "Provide at least 2 tickers, e.g. AAPL,MSFT")
    if len(ticker_list) > 10:
        raise HTTPException(422, "Maximum 10 tickers for correlation.")

    try:
        raw     = yf.download(ticker_list, period="5y",
                              auto_adjust=True, progress=False, timeout=30)
        close   = raw["Close"].dropna()
        log_ret = np.log(close / close.shift(1)).dropna()
        corr    = log_ret.corr().round(4)
    except Exception as e:
        raise HTTPException(503, f"Data download failed: {e}")

    return corr.to_dict()
