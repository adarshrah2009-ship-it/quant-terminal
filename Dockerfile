# Base image with Python runtime
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy requirement definitions first to leverage Docker layer caching
COPY requirements.txt .

# Install system dependencies and Python modules
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt-get/lists/*
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Run Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]