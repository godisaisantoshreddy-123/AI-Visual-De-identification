FROM python:3.10-slim-bullseye

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# ✅ Force Streamlit to use project .streamlit directory
ENV STREAMLIT_CONFIG_DIR=/app/.streamlit
ENV STREAMLIT_HOME=/app

# Expose Streamlit port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
