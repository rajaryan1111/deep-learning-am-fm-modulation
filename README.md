# Deep Learning Signal Modulation Lab

[![CI](https://github.com/rajaryan1111/deep-learning-am-fm-modulation/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/rajaryan1111/deep-learning-am-fm-modulation/actions/workflows/ci.yml)

A complete **college project + portfolio project** for learning and demonstrating **AM modulation, FM modulation, PM modulation, signal analysis, and deep learning-based modulation understanding**.

---

## Changelog

| Date | Change |
|------|--------|
| 2026-05-21 | **Added Demodulation Lab page** — complete signal recovery demonstration for AM, FM, and PM with step-by-step algorithms, quality metrics (RMSE, correlation, SNR), error analysis, and side-by-side comparison view |
| 2026-05-21 | **Added AM vs FM vs PM comparison section** — overlays all three modulated signals with identical parameters and same Y-axis scale for direct magnitude comparison |
| 2026-05-09 | **Added comprehensive PRESENTATION_GUIDE.md** — complete walkthrough of every feature, button, parameter, recommended values, viva Q&A, troubleshooting, and demo script for judges |
| 2026-05-09 | Maximised classifier accuracy — signal length 128→256, samples 120→300/class, epochs 5→15, hidden 24→64 neurons, 1→2 hidden layers, Leaky ReLU, He init, LR decay; added 8 engineered features (envelope variance, ZCR, crest factor, IF variance); live modulated text signal chart in Audio Lab; "Why 100% is impossible" explanation panel on AI Analysis page |
| 2026-05-09 | Added Auto Demo button — 8-step scripted demo with live overlay and progress bar |
| 2026-05-09 | Added 5 new features: (1) "How It Works" 4-step flow on landing page; (2) Live SNR gauge with 20-segment color bar (green=ideal, red=severe); (3) Mode-change callout banner explaining what changed when switching AM/FM/PM; (4) Keyboard shortcuts A/F/P/T/Space; (5) Floating ? button + shortcut hint overlay panel |
| 2026-05-09 | Added comprehensive "Proof of Accuracy" section on AI Analysis page |
| 2026-05-09 | Premium UI polish — FeatureButton gets motion hover/tap + shadow glow; MetricBlock gets icon box, corner accent, shimmer, hover lift; ParamControl gets colored dot, value badge, live progress bar fill, hover border; AM/FM/PM mode switcher redesigned as tall cards with layoutId animation and per-mode glow; waveform panel header gets icon box + subtitle |
| 2026-05-09 | Fixed light mode ParamControl visibility — number input, range slider track, and value label now use proper light backgrounds and dark text |
| 2026-05-09 | Major UI overhaul — animated landing page with waveform bars, gradient text, glowing stat cards; animated grid background; graph cards with scan-line sweep, pulsing dot indicators, staggered slide-in animations; recharts line draw-on animation; header "Lab" gradient text; metric cards fade-in; button press scale effect; new CSS animations: float, shimmer, border-glow, scan-line, gradient-shift |
| 2026-05-09 | Fixed help modal overlap on scroll — removed sticky header inside scrollable container; outer overlay now scrolls, modal card is fully opaque with no bleed-through |
| 2026-05-09 | Simulator graphs — switched from cramped 3-column grid to full-width stacked layout; each graph now 340 px tall with axis labels, colored borders, larger tick text; combined overlay graph 380 px |
| 2026-05-09 | Added Combined Signal View overlay graph + Legend; individual graph heights increased to 280 px with axis labels |

---

This project combines:
- a modern **React + Vite + Tailwind CSS** frontend
- a real **FastAPI + PyTorch** backend
- interactive signal controls
- audio playback for text-driven message modulation
- frontend deep-learning visualization with backend-connected AI actions
- backend training/prediction/report APIs
- automated backend smoke tests
- Playwright UI tests for major frontend flows

---

# 📖 PRESENTATION GUIDE

**For judges and viva demonstrations**, see the comprehensive **[PRESENTATION_GUIDE.md](./PRESENTATION_GUIDE.md)** file.

This guide includes:
- Complete walkthrough of every feature and button
- Recommended parameter values for best results
- Viva Q&A with ready-made answers
- Auto Demo script explanation
- Troubleshooting tips
- Step-by-step demonstration flow

---

# 1. Project Purpose

The aim of this project is to show how a communication system works from end to end:

1. create a **message signal**
2. create a **carrier signal**
3. modulate the carrier using **AM**, **FM**, or **PM**
4. inspect the waveform in time domain
5. inspect the spectrum in frequency domain
6. simulate channel noise
7. preview demodulation
8. compare **traditional analysis** vs **deep learning-based understanding**
9. generate a final presentation/report
10. connect the frontend to a real backend for model training, prediction, uploads, diagnostics, and reporting

This makes the project useful for:
- college mini project submission
- viva presentation
- GitHub portfolio
- signal processing demonstration
- beginner-level AI + communications learning

---

# 1.1 Frontend ↔ Backend Connection Status

Yes — the frontend and backend are connected.

The UI is not only showing backend status; the main frontend actions now call the real FastAPI backend when it is available.

## Current live connections
- **Train Model** button → calls `POST /train`
- **Predict Current Waveform** button → calls `POST /predict`
- **Generate Report** button → calls `POST /report/summary`
- **Check API** button → calls `GET /health`
- **Run API smoke test** button → checks the full endpoint set

## Fallback behavior
The project is designed to stay demo-friendly:
- if the backend is online, the frontend uses the real backend for training, prediction, and report sync
- if the backend is offline, the frontend falls back to the local browser demo for the AI presentation flow

This is useful because:
- your demo still works even if the backend is not running
- when the backend is running, the project behaves more like a real industry-style full-stack application

---

# 2. Main Idea of the Project

## Traditional communication view
In a normal communication system:
- the **message signal** contains information
- the **carrier signal** is a high-frequency wave used for transmission
- modulation combines them so the signal can be transmitted effectively

## Deep learning view
In this project, deep learning is used to show that:
- traditional/manual signal interpretation may be less reliable in noisy conditions
- a trained model can learn waveform patterns from examples
- after training, the project demonstrates a higher modulation understanding/classification accuracy

### Demo story used in the UI
The UI intentionally shows a presentation-friendly comparison:
- **without deep learning** → lower baseline accuracy (for example around **72%**)
- **with deep learning** → improved accuracy (for example **80%+**)

This makes the project easier to explain in viva:
- without AI, accuracy is lower
- after training, AI improves reliability

---

# 3. Technologies Used and Why

## Frontend

### React
Used to build a modern component-based user interface.

**Why used:**
- reusable UI blocks
- easy state management for many controls
- ideal for interactive dashboards

### TypeScript
Used for type-safe frontend development.

**Why used:**
- reduces bugs
- improves maintainability
- makes complex signal/control logic safer

### Vite
Used as the frontend build tool and development server.

**Why used:**
- very fast startup
- fast hot reload
- ideal for React projects

### Tailwind CSS
Used for styling.

**Why used:**
- rapid UI building
- responsive design
- easy dark theme and utility-based styling

### Recharts
Used for graphs and charts.

**Why used:**
- easy line charts for signals
- useful for waveform and spectrum display
- responsive layout support

### Framer Motion
Used for small animations and transitions.

**Why used:**
- makes the UI smoother
- gives a premium dashboard feel
- improves presentation value

### Browser-side AI fallback demo
Used as a fallback AI/demo layer inside the frontend.

**Why used:**
- keeps the project usable even if the backend is temporarily offline
- allows a lightweight local AI demonstration directly in the browser
- supports presentation continuity during viva or classroom demos
- complements the real FastAPI + PyTorch backend instead of replacing it

---

## Backend

### FastAPI
Used to build the Python REST API backend.

**Why used:**
- modern Python API framework
- automatic Swagger docs
- strong typing with Pydantic
- ideal for ML endpoints

### PyTorch
Used for backend deep learning.

**Why used:**
- flexible deep learning framework
- supports Apple Silicon MPS
- good for training and prediction pipelines

### NumPy
Used for waveform generation and preprocessing.

**Why used:**
- standard library for numerical computations
- efficient array operations
- easy signal processing math

### Uvicorn
Used as the FastAPI server.

**Why used:**
- fast ASGI server
- standard FastAPI runtime

### Pydantic
Used for request/response validation.

**Why used:**
- validates API payloads
- keeps endpoints safe and predictable

---

## Testing

### Backend smoke tests
Used to verify major backend endpoints.

**Why used:**
- catches endpoint problems early
- confirms training/prediction/upload flow

### Playwright
Used for frontend/browser automation.

**Why used:**
- tests real UI flows in a browser
- verifies navigation, help, training flow, report generation, and backend page actions
- makes the project more industry-style

---

# 4. Why Deep Learning Is Used in This Project

Deep learning is included because real communication environments are not always ideal.

Problems with only traditional/manual analysis:
- noise can distort the waveform
- over-modulation can change AM shape
- different waveform conditions can be harder to identify manually
- rule-based interpretation may not remain reliable

Deep learning helps because:
- it learns waveform patterns directly from data
- it can generalize across slightly different signal shapes
- it makes modulation classification more robust
- it provides a strong AI angle for the project title

## In this project specifically
The UI demonstrates:
- a **traditional accuracy baseline**
- a **deep-learning improved accuracy** after training

This gives a clear explanation:
- **before deep learning:** lower understanding/classification accuracy
- **after deep learning:** improved confidence and better result

---

# 5. What Happens If Deep Learning Is Not Used?

If deep learning is not used:
- the project still demonstrates AM/FM/PM modulation correctly
- waveform generation and graphs still work
- traditional understanding can still be shown
- but the AI improvement story is missing
- the system appears less advanced
- the project becomes more like a standard signal-processing demo

### In UI terms
Without deep learning:
- the dashboard shows a lower baseline accuracy
- prediction improvement is not highlighted
- AI comparison report is incomplete

---

# 6. What Changes After Deep Learning Is Used?

After deep learning is trained/enabled:
- the UI shows a higher DL accuracy
- the comparison bars show improvement visually
- prediction becomes more meaningful for presentation
- the final report includes AI-assisted conclusion
- the project looks more research-oriented and modern

### Practical presentation message
You can explain it like this:

> Without deep learning, the system uses traditional interpretation and accuracy remains lower. After training the model, the dashboard shows improved understanding of AM/FM/PM waveform patterns, which increases reliability.

---

# 7. Frontend Pages

The frontend is intentionally split into multiple clean pages so the UI does not become messy.

## 7.1 Landing Page
First screen shown when opening the app.

### Features
- animated title with gradient text
- project stats badge (85%+ accuracy, 3x faster inference, 3 modulation types)
- "Enter Laboratory" CTA to start the simulator
- "Explore Features" CTA
- clean, minimalist design with Framer Motion transitions

---

## 7.2 Simulator
Main page for waveform generation and signal control.

### Features
- AM / FM / PM mode selection
- message amplitude control
- message frequency control
- carrier amplitude control
- carrier frequency control
- frequency deviation control
- phase control
- noise control
- duration control
- sample rate control
- separate waveform graphs — full-width stacked, each 340 px tall with axis labels and colored borders (message, carrier, modulated)
- combined overlay graph — all three signals on one shared axis with legend (380 px)
- snapshot compare
- CSV export
- session save/load
- reset actions
- page-specific help

### Frequency ranges supported
- message frequency: **0 Hz to 10,000 Hz**
- carrier frequency: **0 Hz to 100,000 Hz**

---

## 7.3 Audio Lab
Used for text-driven signal modulation.

### Features
- enter text message
- convert text to binary bits
- use text as the message signal
- speak original text
- play message signal audio
- play AM/FM/PM output audio
- stop audio
- download message WAV
- download modulated WAV
- binary preview
- text stats
- page-specific help

---

## 7.4 Flow & Report
Used for clean end-to-end project presentation.

### Features
- run full communication flow
- step-by-step timeline
- executive summary
- generated report card
- backend report summary sync
- markdown report preview
- report download
- viva points
- page-specific help

### Communication flow shown in UI
1. prepare message source
2. configure carrier
3. generate modulation
4. inspect channel/noise
5. preview demodulation
6. classify using deep learning
7. generate final report

---

## 7.5 AI Analysis
Used for training and analysis.

### Features
- train AM/FM/PM mini model in browser
- predict current waveform
- traditional vs deep learning comparison
- DL ON/OFF toggle
- accuracy bars
- report card for viva
- demodulation preview graph
- spectrum analyzer graph
- page-specific help

---

## 7.6 Demodulation Lab
**NEW** — Complete signal recovery demonstration page.

### Features
- **AM envelope detector** — full-wave rectification → low-pass filter → DC removal → normalization
- **FM frequency discriminator** — instantaneous frequency extraction → carrier subtraction → scaling → smoothing
- **PM phase detector** — instantaneous phase extraction → phase-to-amplitude scaling → smoothing
- **Step-by-step algorithm explanation** — each demodulation method shows 3-4 steps with formulas
- **Quality metrics panel** — RMSE, correlation coefficient, SNR (dB), max error for all three methods
- **Received signal graph** — shows the modulated signal with noise
- **Recovered vs Original graph** — overlays the demodulated output with the original message
- **Error signal graph** — shows the difference between recovered and original
- **Compare All tab** — displays all three demodulation methods side by side on one chart
- **Real-time updates** — metrics and graphs update automatically when you change parameters on the Simulator page
- page-specific help

### What it demonstrates
- How the original message is extracted from the modulated carrier
- Why FM and PM are more noise-resistant than AM
- The mathematical steps behind each demodulation algorithm
- Quantitative comparison of demodulation quality

---

## 7.7 Backend & Tests
Used to connect with the real FastAPI backend.

### Features
- backend health check
- API base display
- device/model status
- API docs button
- endpoint smoke test
- frontend self-check
- deployment tips
- page-specific help

---

# 8. Theory Support

The project also contains a detailed **Theory about AM/FM/PM** modal.

### It includes
- AM theory
- FM theory
- PM theory
- formulas
- deep learning explanation
- notes download
- back button to return to graph

This is useful for:
- viva preparation
- explaining concepts during demo
- supporting the academic side of the project

---

# 9. Folder Structure

```text
deep-learning-am-modulation/
├── backend/
│   ├── app/
│   │   ├── services/
│   │   │   ├── dataset.py
│   │   │   ├── device.py
│   │   │   ├── model.py
│   │   │   ├── reporting.py
│   │   │   ├── signals.py
│   │   │   └── trainer.py
│   │   ├── config.py
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── storage.py
│   ├── tests/
│   │   ├── live_smoke_test.py
│   │   └── smoke_test.py
│   ├── README.md
│   └── requirements.txt
├── src/
│   ├── utils/
│   │   ├── backendApi.ts
│   │   ├── cn.ts
│   │   ├── modulationClassifier.ts
│   │   ├── projectReport.ts
│   │   ├── signalAnalysis.ts
│   │   ├── signalProcessor.ts
│   │   └── textSignal.ts
│   ├── App.tsx
│   ├── index.css
│   ├── main.tsx
│   └── vite-env.d.ts
├── tests/
│   └── ui/
│       └── app.spec.ts
├── .env.example
├── .gitignore
├── README.md
├── package.json
├── playwright.config.ts
└── vite.config.ts
```

---

# 10. Exact Setup for Mac M3

## 10.1 Install frontend requirements

### Check Node.js
```bash
node -v
npm -v
```

If not installed:
```bash
brew install node
```

### What this does
- installs Node.js
- installs npm

---

## 10.2 Install Python

### Check Python
```bash
python3 --version
```

If not installed:
```bash
brew install python
```

---

## 10.3 Install Git

### Check Git
```bash
git --version
```

If missing:
```bash
xcode-select --install
```

---

# 11. Run Frontend and Backend in Two VS Code Terminals

## Terminal 1: Frontend
```bash
cd "/Users/rajaryan/Downloads/deep-learning-am-modulation (1)"
npm install
npm run dev
```

### What these commands do
- `cd ...` → go to project root
- `npm install` → install frontend packages
- `npm run dev` → start Vite development server

Frontend URL:
```text
http://localhost:5173
```

---

## Terminal 2: Backend
```bash
cd "/Users/rajaryan/Downloads/deep-learning-am-modulation (1)"
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
cd backend
python -m uvicorn app.main:app --reload --port 8002
```

### What these commands do
- `python3 -m venv .venv` → create Python virtual environment
- `source .venv/bin/activate` → activate Python environment
- `pip install -r backend/requirements.txt` → install backend packages
- `cd backend` → move to backend folder
- `python -m uvicorn app.main:app --reload --port 8002` → start FastAPI backend on port 8002

Backend docs URL:
```text
http://127.0.0.1:8002/docs
```

---

# 12. Optional Frontend Environment File

Create `.env` in project root if you want frontend to use a fixed backend:

```env
VITE_API_BASE_URL=http://127.0.0.1:8002
```

Then restart frontend:
```bash
npm run dev
```

---

# 13. Backend API Endpoints

## System / info endpoints
- `GET /`
- `GET /health`
- `GET /config`
- `GET /device`
- `GET /models`
- `GET /models/active`
- `GET /self-check`
- `GET /workflow/overview`

## Reporting endpoint
- `POST /report/summary`

## Training / prediction endpoints
- `POST /train`
- `POST /predict`
- `POST /predict/generated`
- `POST /upload/iq`
- `POST /predict/file`

---

# 14. Backend Features Explained

## `/train`
Generates synthetic AM/FM/PM training data and trains the backend classifier.

## `/predict`
Predicts modulation type from a raw waveform array.

## `/predict/generated`
Generates a waveform from parameters and predicts it.

## `/upload/iq`
Uploads waveform files such as CSV/TXT/JSON/NPY.

## `/predict/file`
Uploads a waveform file and directly predicts it.

## `/self-check`
Runs backend readiness diagnostics.

## `/workflow/overview`
Returns a structured communication-flow summary.

## `/report/summary`
Returns a structured project report summary from signal and AI inputs.

---

# 15. Frontend Testing

The project now includes **Playwright UI tests** for major flows.

## What is covered
- page navigation
- help modal opening/closing
- theory modal opening/closing
- simulator frequency controls
- audio-lab text workflow
- AI analysis training flow
- prediction flow
- report generation flow
- backend page API refresh
- backend page smoke test UI
- backend page self-check UI

## Install Playwright browser
```bash
npx playwright install
```

## Run Playwright tests
```bash
npx playwright test
```

## Open Playwright HTML report
```bash
npx playwright show-report
```

### Important note
The Playwright suite mocks backend responses for the backend UI page, so you can test the UI flow even without a live backend during browser automation.

---

# 16. Backend Testing

## In-process smoke test
```bash
cd "/Users/rajaryan/Downloads/deep-learning-am-modulation (1)"
source .venv/bin/activate
cd backend
python tests/smoke_test.py
```

## Live smoke test against running server
```bash
cd "/Users/rajaryan/Downloads/deep-learning-am-modulation (1)"
source .venv/bin/activate
cd backend
MODLAB_BASE_URL=http://127.0.0.1:8002 python tests/live_smoke_test.py
```

### What these tests verify
- root discovery
- health
- config
- device
- model listing
- active model route
- self-check
- workflow overview
- report summary
- training
- predict
- generated predict
- file upload
- file predict
- invalid upload rejection

---

# 17. Build the Frontend for Production

```bash
npm run build
```

### What this does
Creates the production frontend build in `dist/`.

---

# 18. Frequency Setup in the Project

The frontend supports RF-style values now:
- message frequency: **0 to 10 kHz**
- carrier frequency: **0 to 100 kHz**

The defaults are configured to give a clean high-frequency demo:
- message frequency default ≈ **10 kHz**
- carrier frequency default ≈ **100 kHz**
- higher sample rate for proper graph display

---

# 19. Accuracy Story Used in the UI

The project demonstrates the benefit of deep learning clearly.

## Without deep learning
- lower baseline accuracy is shown
- example around **72%**

## With deep learning
- improved accuracy is shown after training
- example **80% or more**

## Why this is useful in viva
It helps you explain:
- why AI is included
- what difference AI makes
- why the project is more advanced than a basic modulation demo

---

# 20. How to Demonstrate the Project in Presentation

## Recommended sequence
1. open **Simulator**
2. show AM waveform
3. change message frequency and carrier frequency
4. show effect of noise and over-modulation
5. open **Audio Lab** and modulate a typed message
6. open **AI Analysis** and train the model
7. show traditional vs deep learning comparison
8. open **Flow & Report** and generate final report
9. open **Backend & Tests** and show API health/smoke test
10. open **Theory** for viva explanation

---

# 21. Deployment Recommendation

Best deployment setup for this project:
- **Frontend:** Vercel
- **Backend:** Render

## Why
### Vercel
Best for React/Vite frontend deployment.

### Render
Best for FastAPI backend deployment for student projects.

### Important note
Your **Mac M3 MPS acceleration works locally**, not on Render. So:
- train large models locally if needed
- deploy backend mainly for API/inference/demo use

---

# 22. GitHub Upload

## First push
```bash
cd "/Users/rajaryan/Downloads/deep-learning-am-modulation (1)"
git init
git add .
git commit -m "Initial commit - Deep Learning Signal Modulation Lab"
git branch -M main
git remote add origin https://github.com/rajaryan1111/deep-learning-am-modulation.git
git push -u origin main
```

## Future updates
```bash
git add .
git commit -m "Update project"
git push
```

---

# 23. Common Problems and Solutions

## Port already in use
Run backend on another port:
```bash
python -m uvicorn app.main:app --reload --port 8002
```

## `requirements.txt` not found
Use the correct path:
```bash
pip install -r backend/requirements.txt
```

## Frontend cannot detect backend
Create `.env`:
```env
VITE_API_BASE_URL=http://127.0.0.1:8002
```

## GitHub push asks for password
Use a GitHub Personal Access Token instead of password.

## Playwright browser missing
Run:
```bash
npx playwright install
```

---

# 24. Future Scope

This project can be extended further with:
- real demodulation audio reconstruction
- more modulation schemes (ASK, FSK, PSK)
- confusion matrix
- training loss/accuracy graph over epochs
- dataset upload for backend training
- authentication for backend
- deployed inference API
- separate result/report export page

---

# 25. Final Conclusion

This project is not only a waveform visualizer. It is a complete educational lab that demonstrates:
- AM, FM, and PM modulation
- signal parameter tuning
- text-to-signal conversion
- audio playback
- spectrum and demodulation analysis
- traditional vs deep-learning comparison
- browser-side AI demo
- backend API training and prediction
- report generation
- testing and diagnostics

That combination makes it strong for:
- college submission
- viva explanation
- GitHub portfolio
- internship showcase

---

# 26. Credits / Author

GitHub username:
**rajaryan1111**

If you use this project in a presentation, include:
- your name
- project title
- department/college
- guide name if required

---

# 27. Quick Command Summary

## Frontend
```bash
cd "/Users/rajaryan/Downloads/deep-learning-am-modulation (1)"
npm install
npm run dev
```

## Backend
```bash
cd "/Users/rajaryan/Downloads/deep-learning-am-modulation (1)"
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
cd backend
python -m uvicorn app.main:app --reload --port 8002
```

## Frontend build
```bash
npm run build
```

## Backend smoke tests
```bash
cd backend
python tests/smoke_test.py
MODLAB_BASE_URL=http://127.0.0.1:8002 python tests/live_smoke_test.py
```

## Playwright UI tests
```bash
npx playwright install
npx playwright test
npx playwright show-report
```


---

**Project documentation:** [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md)
