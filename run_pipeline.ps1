$ErrorActionPreference = "Stop"
Write-Host "Starting predictive maintenance pipeline..." -ForegroundColor Cyan
python src/train_model.py
if ($LASTEXITCODE -ne 0) {
    throw "Training failed."
}
python src/predict_metropt3.py
if ($LASTEXITCODE -ne 0) {
    throw "Prediction failed."
}
Write-Host "Pipeline completed successfully." -ForegroundColor Green
