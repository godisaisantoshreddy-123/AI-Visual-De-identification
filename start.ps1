# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# Detect app type and start
if (Get-Content main.py | Select-String "streamlit") {
    Write-Host "Starting Streamlit app..."
    streamlit run main.py --server.port $env:PORT --server.address 0.0.0.0
} elseif (Get-Content main.py | Select-String "Flask") {
    Write-Host "Starting Flask app..."
    gunicorn main:app --bind 0.0.0.0:$env:PORT
} elseif (Get-Content main.py | Select-String "FastAPI") {
    Write-Host "Starting FastAPI app..."
    gunicorn -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:$env:PORT
} else {
    Write-Host "Could not detect app type. Please start manually."
}
