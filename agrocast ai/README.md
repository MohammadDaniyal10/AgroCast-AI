# AgroCast AI - Setup Guide (Original UI, Portable)

Two small Flask apps, kept exactly as your original UI:
- Price Prediction app (port 5000): `Crop-Price-Prediction-Using-Random-Forest-main/src`
- Crop Recommendation app (port 5001): `Crop_Recommendation-main`

## 1) Requirements
- Python 3.12+ (3.10/3.11 also OK)
- Internet access for `pip install`

## 2) Start Price Prediction (5000)
```powershell
cd "C:\faiz  project.01\Crop-Price-Prediction-Using-Random-Forest-main\src"
python -m venv .venv
.venv\Scripts\pip install --upgrade pip
.venv\Scripts\pip install -r requirements.txt
$env:MONGO_URL="mongodb://localhost:27017/cropdb"
.venv\Scripts\python app.py
```
Open: `http://127.0.0.1:5000/`

## 3) Start Crop Recommendation (5001)
```powershell
cd "C:\faiz  project.01\Crop_Recommendation-main"
python -m venv .venv
.venv\Scripts\pip install --upgrade pip
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python app.py
```
Open: `http://127.0.0.1:5001/`

## 4) One-command start (optional)
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& "C:\faiz  project.01\scripts\start_all.ps1"
```

## 5) Troubleshooting
- 127.0.0.1 refused:
  - Keep both PowerShell servers open; allow Firewall
  - `Stop-Process -Name python -ErrorAction SilentlyContinue` then restart
- Version warnings:
  - Price models load fine on 1.7 with warnings
  - Recommendation is pinned to 1.3.2 in its requirements.txt

## 6) Deploy on Render (production)
This repo includes a root `render.yaml` defining two services:
- `agrocast-price` (builds with `Crop-Price-Prediction-Using-Random-Forest-main/src/requirements.txt`)
- `agrocast-recommend` (builds with `Crop_Recommendation-main/requirements.txt`)

Steps:
1) Push this folder to GitHub
2) Render → New Web Service → connect repo → Render reads `render.yaml`

## 7) Structure
- `Crop-Price-Prediction-Using-Random-Forest-main/src/`: app.py, requirements.txt, Procfile, templates/, static/
- `Crop_Recommendation-main/`: app.py, requirements.txt, Procfile, templates/
- `scripts/start_all.ps1`: optional helper
- `render.yaml`: deploy both services
