# ==============================
# Run SDXL Flask Web UI (with auto browser open)
# ==============================

Write-Host "🔹 Activating virtual environment..."
. .\.venv\Scripts\Activate.ps1

Write-Host "🔹 Starting Flask app..."
Start-Process "http://localhost:5000"
python app.py
