import { useState, useEffect } from "react";

const API_BASE = "http://localhost:5000/api";

function sparkPath(prices, w = 400, h = 110) {
  if (!prices.length) return "";
  const min = Math.min(...prices);
  const max = Math.max(...prices);
  const range = max - min || 1;
  const pts = prices.map((p, i) => {
    const x = (i / (prices.length - 1)) * w;
    const y = h - ((p - min) / range) * h;
    return `${x},${y}`;
  });
  return "M " + pts.join(" L ");
}

function MetricCard({ label, value, sub }) {
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      {sub && <div className="metric-sub">{sub}</div>}
    </div>
  );
}

function Loader({ message }) {
  return (
    <div className="loader-wrap">
      <div className="loader-ring" />
      <p className="loader-text">{message || "Fetching live prices & running models…"}</p>
    </div>
  );
}

export default function App() {
  const [forecast, setForecast]   = useState(null);
  const [metrics, setMetrics]     = useState(null);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState("");
  const [lastPrice, setLastPrice] = useState(null);

  const runAutoForecast = async () => {
    setError("");
    setLoading(true);
    setForecast(null);
    try {
      const res  = await fetch(`${API_BASE}/forecast/auto`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Forecast failed");
      setForecast(data);
      setLastPrice(data.last_actual);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetch(`${API_BASE}/metrics`)
      .then(r => r.json())
      .then(d => setMetrics(d))
      .catch(() => {});

    // Auto-run on page load
    runAutoForecast();
  }, []);

  const prices      = forecast?.prices ?? [];
  const high        = prices.length ? Math.max(...prices) : 0;
  const low         = prices.length ? Math.min(...prices) : 0;
  const netChange   = prices.length ? prices[prices.length - 1] - prices[0] : 0;
  const netPct      = prices.length ? ((netChange / prices[0]) * 100).toFixed(2) : 0;
  const trending    = netChange >= 0;

  return (
    <div className="shell">
      <div className="bg-grid" />

      <header className="topbar">
        <div className="topbar-left">
          <div className="logo-mark">◈</div>
          <div>
            <div className="brand">OilPulse</div>
            <div className="brand-sub">WTI CRUDE · LIVE ENSEMBLE FORECAST</div>
          </div>
        </div>
        <div className="topbar-right">
          {lastPrice && (
            <div className="live-price">
              <span className="live-dot" />
              <span className="live-label">Last Close</span>
              <span className="live-val">${lastPrice.toFixed(2)}</span>
            </div>
          )}
          <button className="btn-refresh" onClick={runAutoForecast} disabled={loading}>
            {loading ? "…" : "⟳ Refresh"}
          </button>
        </div>
      </header>

      <main className="content">

        {/* Model performance */}
        {metrics?.ensemble_test && (
          <section className="panel">
            <div className="panel-title">🧠 Ensemble Model Performance</div>
            <div className="metrics-grid">
              <MetricCard label="MAE"  value={metrics.ensemble_test.MAE}          sub="Mean Abs. Error" />
              <MetricCard label="RMSE" value={metrics.ensemble_test.RMSE}         sub="Root Mean Sq." />
              <MetricCard label="R²"   value={metrics.ensemble_test.R2}           sub="Coeff. of Det." />
              <MetricCard label="MAPE" value={`${metrics.ensemble_test.MAPE}%`}   sub="Mean Abs. Pct." />
            </div>
            <p className="model-tag">{forecast?.model ?? "LSTM · CNN · LightGBM · Ridge Meta-Model"}</p>
          </section>
        )}

        {loading && <Loader />}

        {error && (
          <div className="error-box">
            ⚠ {error}
            {error.includes("models") && (
              <div style={{ marginTop: 8, fontSize: 12 }}>
                Make sure your <strong>models/</strong> folder is in the project root and contains all 7 files.
              </div>
            )}
          </div>
        )}

        {forecast && !loading && (
          <section className="panel forecast-panel">
            <div className="forecast-header">
              <div>
                <div className="panel-title">10-Day WTI Crude Oil Forecast</div>
                <div className="gen-time">
                  {forecast.source} · Generated {new Date(forecast.generated_at).toLocaleString()}
                </div>
              </div>
              <div className={`trend-badge ${trending ? "up" : "down"}`}>
                {trending ? "↗" : "↘"} {trending ? "+" : ""}{netPct}%
              </div>
            </div>

            {/* Sparkline */}
            <div className="chart-wrap">
              <div className="chart-y-labels">
                <span>${Math.max(...prices).toFixed(0)}</span>
                <span>${Math.min(...prices).toFixed(0)}</span>
              </div>
              <div className="chart-area">
                <svg viewBox="-5 -5 410 120" className="sparkline-svg" preserveAspectRatio="none">
                  <defs>
                    <linearGradient id="fillGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#f0a500" stopOpacity="0.3" />
                      <stop offset="100%" stopColor="#f0a500" stopOpacity="0" />
                    </linearGradient>
                  </defs>
                  <path d={sparkPath(prices) + " L 400,110 L 0,110 Z"} fill="url(#fillGrad)" />
                  <path d={sparkPath(prices)} fill="none" stroke="#f0a500" strokeWidth="2.5"
                        strokeLinejoin="round" strokeLinecap="round" />
                  {prices.map((p, i) => {
                    const min = Math.min(...prices), max = Math.max(...prices), range = max - min || 1;
                    const x = (i / (prices.length - 1)) * 400;
                    const y = 110 - ((p - min) / range) * 110;
                    return (
                      <g key={i}>
                        <circle cx={x} cy={y} r="5" fill="#0d0f14" stroke="#f0a500" strokeWidth="2" />
                        <text x={x} y={y - 10} textAnchor="middle"
                              style={{ fontSize: 9, fill: "#9ca3af", fontFamily: "monospace" }}>
                          ${p.toFixed(1)}
                        </text>
                      </g>
                    );
                  })}
                </svg>
                <div className="chart-dates">
                  {forecast.dates.map((d, i) => (
                    <span key={i} className="chart-label">{d.slice(5)}</span>
                  ))}
                </div>
              </div>
            </div>

            {/* Summary */}
            <div className="summary-row">
              <MetricCard label="Day 1 Forecast" value={`$${prices[0]?.toFixed(2)}`} />
              <MetricCard label="10-Day High"    value={`$${high.toFixed(2)}`} />
              <MetricCard label="10-Day Low"     value={`$${low.toFixed(2)}`} />
              <MetricCard
                label="Net Change"
                value={`${netChange >= 0 ? "+" : ""}$${netChange.toFixed(2)}`}
                sub={`${netChange >= 0 ? "+" : ""}${netPct}%`}
              />
            </div>

            {/* Day-by-day table */}
            <div className="forecast-table">
              <div className="table-head">
                <span>Day</span><span>Date</span>
                <span>Price (USD/bbl)</span><span>vs Day 1</span>
              </div>
              {prices.map((p, i) => {
                const delta = p - prices[0];
                const pct   = ((delta / prices[0]) * 100).toFixed(2);
                const up    = delta >= 0;
                return (
                  <div key={i} className={`forecast-row ${i % 2 === 0 ? "even" : ""}`}
                       style={{ animationDelay: `${i * 50}ms` }}>
                    <span className="row-day">Day {i + 1}</span>
                    <span className="row-date">{forecast.dates[i]}</span>
                    <span className="row-price">${p.toFixed(2)}</span>
                    <span className={`row-delta ${up ? "up" : "down"}`}>
                      {up ? "+" : ""}{delta.toFixed(2)} ({up ? "+" : ""}{pct}%)
                    </span>
                  </div>
                );
              })}
            </div>
          </section>
        )}
      </main>

      <footer className="footer">
        OilPulse · Prices sourced from Yahoo Finance · For informational purposes only · Not financial advice
      </footer>
    </div>
  );
}
