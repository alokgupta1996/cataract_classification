# Use Python 3.9 slim image for smaller size
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code except the training_data folder
COPY . .

# Create necessary directories
RUN mkdir -p customer_data models logs temp

# Expose port
EXPOSE 8001

# Set environment variables
ENV PYTHONPATH=/app
ENV RAY_DISABLE_IMPORT_WARNING=1

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run the API with uvicorn and multiple workers for concurrent requests
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"] 