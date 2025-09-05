FROM python:3.10-slim-bullseye

# Set working directory
WORKDIR /app

# Copy files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ✅ Tell Streamlit where to find config
ENV STREAMLIT_CONFIG_DIR=/app/.streamlit

# Run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
