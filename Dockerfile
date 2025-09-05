FROM python:3.10-slim-bullseye

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Make start.sh executable
RUN chmod +x start.sh

# Expose Streamlit port
EXPOSE 8501

# Run wrapper script instead of calling Streamlit directly
CMD ["./start.sh"]
