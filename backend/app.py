"""
WTI Crude Oil 10-Day Forecast API
Ensemble: LSTM + CNN + LightGBM + Ridge meta-model
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle
import json
import os
import urllib.request
from datetime import datetime, timedelta

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

app = Flask(__name__)
CORS(app)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
_cache = {}

# ── Live price fetcher ────────────────────────────────────────
def fetch_live_wti_prices(n=120):
    end   = int(datetime.utcnow().timestamp())
    start = int((datetime.utcnow() - timedelta(days=n * 2)).timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/CL=F"
        f"?interval=1d&period1={start}&period2={end}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())

    result   = data["chart"]["result"][0]
    quote    = result["indicators"]["quote"][0]
    closes   = quote["close"]
    opens    = quote["open"]
    highs    = quote["high"]
    lows     = quote["low"]

    rows = []
    for i in range(len(closes)):
        if closes[i] is not None:
            rows.append({
                "Price": closes[i],
                "Open":  opens[i]  if opens[i]  else closes[i],
                "High":  highs[i]  if highs[i]  else closes[i],
                "Low":   lows[i]   if lows[i]   else closes[i],
            })

    return rows[-n:]


# ── Feature builder ───────────────────────────────────────────
def build_features(rows):
    """
    Given a list of dicts with Price/Open/High/Low,
    compute all 11 features the model expects:
    ['Price','Open','High','Low','Change %','ma_5','ma_20',
     'RSI','momentum','volatility','price_change']
    """
    df = pd.DataFrame(rows)

    df["Change %"]    = df["Price"].pct_change() * 100
    df["ma_5"]        = df["Price"].rolling(5).mean()
    df["ma_20"]       = df["Price"].rolling(20).mean()
    df["momentum"]    = df["Price"] - df["Price"].shift(10)
    df["volatility"]  = df["Price"].rolling(10).std()
    df["price_change"]= df["Price"].diff()

    # RSI
    delta  = df["Price"].diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    rs     = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    df = df.bfill().ffill().fillna(0)

    feature_order = ['Price','Open','High','Low','Change %',
                     'ma_5','ma_20','RSI','momentum','volatility','price_change']
    return df[feature_order].values


# ── Model loader ──────────────────────────────────────────────
def load_artifacts():
    if _cache:
        return _cache

    config_path = os.path.join(MODELS_DIR, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError("Models not found. Copy your models/ folder first.")

    with open(config_path) as f:
        _cache["config"] = json.load(f)

    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
        _cache["scaler"] = pickle.load(f)

    with open(os.path.join(MODELS_DIR, "lgb_model.pkl"), "rb") as f:
        _cache["lgb"] = pickle.load(f)

    with open(os.path.join(MODELS_DIR, "ridge_meta_model.pkl"), "rb") as f:
        _cache["ridge"] = pickle.load(f)

    if TF_AVAILABLE:
        try:
            _cache["lstm"] = tf.keras.models.load_model(
                os.path.join(MODELS_DIR, "lstm_model.h5"), compile=False
            )
            _cache["cnn"] = tf.keras.models.load_model(
                os.path.join(MODELS_DIR, "cnn_model.h5"), compile=False
            )
        except Exception as e:
            print(f"TF models failed to load: {e}")

    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            _cache["metrics"] = json.load(f)

    return _cache


# ── Forecast engine ───────────────────────────────────────────
def run_forecast(rows: list) -> dict:
    arts       = load_artifacts()
    cfg        = arts["config"]
    scaler     = arts["scaler"]
    seq_len    = cfg["seq_len"]        # 60
    horizon    = cfg["forecast_horizon"]  # 10
    price_idx  = cfg["price_index"]    # 0 (Price is first)
    n_features = len(cfg["total_fet"]) # 11

    # Build full feature matrix
    features = build_features(rows)    # shape (N, 11)

    if len(features) < seq_len:
        raise ValueError(f"Need at least {seq_len} rows of data, got {len(features)}.")

    features   = features[-seq_len:]   # take last 60 rows
    last_price = float(features[-1, price_idx])

    scaled  = scaler.transform(features)          # (60, 11)
    x_input = scaled[np.newaxis, :, :]            # (1, 60, 11)
    x_flat  = x_input.reshape(1, -1)              # (1, 660)

    if TF_AVAILABLE and "lstm" in arts and "cnn" in arts:
        lstm_pred = arts["lstm"].predict(x_input, verbose=0).reshape(1, horizon, n_features)
        cnn_pred  = arts["cnn"].predict(x_input, verbose=0).reshape(1, horizon, n_features)
    else:
        # Momentum fallback using real feature data
        returns  = np.diff(features[:, price_idx]) / features[:-1, price_idx]
        avg_ret  = float(np.mean(returns[-10:]))
        fallback = np.tile(features[-1:], (horizon, 1)).copy()
        p = last_price
        for t in range(horizon):
            p = p * (1 + avg_ret * (0.9 ** t))
            fallback[t, price_idx] = p
        fallback_scaled = scaler.transform(fallback)[np.newaxis]
        lstm_pred = fallback_scaled
        cnn_pred  = fallback_scaled

    lgb_pred = arts["lgb"].predict(x_flat).reshape(1, horizon, n_features)

    X_meta = np.concatenate([
        lstm_pred.reshape(1, -1),
        cnn_pred.reshape(1, -1),
        lgb_pred.reshape(1, -1)
    ], axis=1)

    ensemble_scaled = arts["ridge"].predict(X_meta).reshape(horizon, n_features)
    ensemble_orig   = scaler.inverse_transform(ensemble_scaled)
    forecast_prices = ensemble_orig[:, price_idx].tolist()

    # Sanity clamp — max 25% move over 10 days
    clamped = []
    prev = last_price
    for p in forecast_prices:
        if abs(p - last_price) / last_price > 0.25:
            p = prev * (1 + np.sign(p - prev) * 0.01)
        clamped.append(round(p, 2))
        prev = p

    # Business day dates
    dates, d = [], datetime.today() + timedelta(days=1)
    while len(dates) < horizon:
        if d.weekday() < 5:
            dates.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)

    return {
        "dates":        dates,
        "prices":       clamped,
        "last_actual":  round(last_price, 2),
        "model":        "LSTM + CNN + LightGBM + Ridge Ensemble" if (TF_AVAILABLE and "lstm" in arts) else "LightGBM + Ridge Ensemble",
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


# ── Routes ────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    models_ready = os.path.exists(os.path.join(MODELS_DIR, "config.json"))
    return jsonify({"status": "ok", "models_loaded": models_ready, "tf_available": TF_AVAILABLE})

@app.route("/api/metrics", methods=["GET"])
def metrics():
    try:
        arts = load_artifacts()
        return jsonify(arts.get("metrics", {}))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/forecast/auto", methods=["GET"])
def forecast_auto():
    try:
        rows = fetch_live_wti_prices(120)
    except Exception as e:
        return jsonify({"error": f"Could not fetch live prices: {e}"}), 503
    try:
        result = run_forecast(rows)
        result["source"] = "Yahoo Finance (live)"
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/forecast", methods=["POST"])
def forecast():
    body = request.get_json(force=True, silent=True) or {}
    rows = body.get("rows", [])
    if not rows:
        # Accept plain prices list too
        prices = body.get("prices", [])
        if prices:
            rows = [{"Price": p, "Open": p, "High": p, "Low": p} for p in prices]
    if not rows:
        return jsonify({"error": "Provide 'rows' or 'prices' in request body."}), 400
    try:
        return jsonify(run_forecast(rows))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
