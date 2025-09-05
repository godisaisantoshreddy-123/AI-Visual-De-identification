# Base image with Python
FROM python:3.10-slim-bullseye

# Prevents Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Install system dependencies (for OpenCV, Tesseract, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libgl1 \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Add Streamlit config
RUN mkdir -p /app/.streamlit
COPY .streamlit /app/.streamlit

# Expose port (Streamlit default is 8501)
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
