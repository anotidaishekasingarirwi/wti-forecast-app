# OilPulse — WTI Crude Oil 10-Day Forecast App

> Ensemble model (LSTM + CNN + LightGBM + Ridge) wrapped in a Flask API + React frontend.

---

## Project Structure

```
wti-forecast-app/
├── backend/
│   ├── app.py              ← Flask API
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       └── index.css
├── models/                 ← Copy your saved model files here
│   ├── lstm_model.h5
│   ├── cnn_model.h5
│   ├── lgb_model.pkl
│   ├── ridge_meta_model.pkl
│   ├── scaler.pkl
│   ├── config.json
│   └── metrics.json
└── README.md
```

---

## Step 1 — Copy Your Models

After running the notebook, copy the `models/` folder it generated into the root of this project:

```bash
cp -r /path/to/notebook/models ./models
```

---

## Step 2 — Open in VS Code

```bash
code wti-forecast-app
```

Install the recommended VS Code extensions:
- **Python** (ms-python.python)
- **ES7+ React/Redux/React-Native** (dsznajder.es7-react-js-snippets)
- **REST Client** (humao.rest-client) — to test API routes
- **Thunder Client** (rangav.vscode-thunder-client) — optional API GUI

---

## Step 3 — Backend Setup

Open the **VS Code integrated terminal** (`Ctrl+`` ` or `Terminal → New Terminal`).

```bash
# Navigate to backend
cd backend

# Create and activate virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate

# macOS / Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the Flask dev server
python app.py
```

The API will run on **http://localhost:5000**

### Test the API from VS Code

Open `Thunder Client` sidebar or use the REST Client extension. Quick test:

```
GET http://localhost:5000/api/health
GET http://localhost:5000/api/forecast/demo
```

---

## Step 4 — Frontend Setup

Open a **second terminal tab** in VS Code (`+` icon in the terminal panel):

```bash
cd frontend
npm install
npm run dev
```

The React app opens at **http://localhost:3000**

> The Vite proxy (`vite.config.js`) forwards all `/api/*` requests to Flask automatically — no CORS issues.

---

## Step 5 — Use the App

1. Open **http://localhost:3000** in your browser
2. Paste recent WTI daily closing prices (one per line or comma-separated), most-recent **last**
3. Click **Generate Forecast** → the ensemble model returns a 10-day prediction
4. Or click **Demo Mode** to see the app in action without entering data

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Server + model status |
| GET | `/api/metrics` | Ensemble test-set metrics |
| POST | `/api/forecast` | Real forecast (body: `{"prices": [...]}`) |
| GET | `/api/forecast/demo` | Demo forecast (no body needed) |

### POST /api/forecast — example body

```json
{
  "prices": [88.49, 90.32, 91.38, 94.48, 93.31, 99.64, 102.60, 102.80, 102.83, 101.28]
}
```

---

## Deployment (Production)

### Option A — Local production build

```bash
# Build the React frontend
cd frontend && npm run build

# Serve static files from Flask (copy dist/ to backend/static/)
cp -r dist/* ../backend/static/

# Run with Gunicorn
cd ../backend
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Option B — Render.com (free tier, recommended)

1. Push project to GitHub
2. Go to **render.com → New Web Service**
3. Connect your GitHub repo
4. **Build command**: `pip install -r backend/requirements.txt`
5. **Start command**: `gunicorn backend.app:app`
6. Set environment variable: `FLASK_ENV=production`

### Option C — Railway.app

```bash
# Install Railway CLI
npm install -g @railway/cli
railway login
railway init
railway up
```

### Option D — Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY backend/ .
COPY models/ ./models/
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "app:app"]
```

```bash
docker build -t oilpulse .
docker run -p 5000:5000 oilpulse
```

---

## VS Code Launch Config (Debug)

Create `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Flask API",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/backend/app.py",
      "env": { "FLASK_ENV": "development" },
      "jinja": true
    }
  ]
}
```

Press **F5** to start the Flask server in debug mode with breakpoints.

---

## Notes

- The app runs in **demo mode** if model files aren't present — it uses a lightweight extrapolation
- All model files are loaded once and cached in memory for fast inference
- Forecasts skip weekends (WTI trades Mon–Fri)
- The scaler from training is used to normalize inputs — always use the same `scaler.pkl`
