# Start Price Prediction backend (port 5000)
$pricePath = "C:\faiz  project.01\Crop-Price-Prediction-Using-Random-Forest-main\src"
$pricePy = Join-Path $pricePath ".venv\Scripts\python.exe"
if (!(Test-Path $pricePy)) {
  python -m venv (Join-Path $pricePath ".venv")
}
& (Join-Path $pricePath ".venv\Scripts\pip.exe") install -r (Join-Path $pricePath "requirements.txt")
$env:MONGO_URL = "mongodb://localhost:27017/cropdb"
Start-Process -FilePath $pricePy -ArgumentList (Join-Path $pricePath "app.py") -WorkingDirectory $pricePath -WindowStyle Normal

# Start Crop Recommendation backend (port 5001) with pinned requirements
$recPath = "C:\faiz  project.01\Crop_Recommendation-main"
$recPy = Join-Path $recPath ".venv\Scripts\python.exe"
if (!(Test-Path $recPy)) {
  python -m venv (Join-Path $recPath ".venv")
}
if (Test-Path (Join-Path $recPath "requirements.txt")) {
  & (Join-Path $recPath ".venv\Scripts\pip.exe") install -r (Join-Path $recPath "requirements.txt")
} else {
  & (Join-Path $recPath ".venv\Scripts\pip.exe") install "scikit-learn==1.3.2" "numpy==1.26.4" "scipy==1.11.4" flask pandas
}
Start-Process -FilePath $recPy -ArgumentList (Join-Path $recPath "app.py") -WorkingDirectory $recPath -WindowStyle Normal

Start-Sleep -Seconds 2
Start-Process "http://127.0.0.1:5000/"

