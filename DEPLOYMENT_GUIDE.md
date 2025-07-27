# 🐳 Docker Deployment Guide

## 📋 **Overview**

This guide explains how to deploy the JIVI Cataract Classification API using Docker with proper async support for handling concurrent requests.

## 🚀 **Quick Start**

### **1. Build and Run with Docker Compose (Recommended)**

```bash
# Build and start the service
docker-compose up --build

# Run in background
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop the service
docker-compose down
```

### **2. Build and Run with Docker**

```bash
# Build the image
docker build -t cataract-api .

# Run the container
docker run -d \
  --name cataract-api \
  -p 8001:8001 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/customer_data:/app/customer_data \
  --memory=4g \
  --cpus=2.0 \
  cataract-api
```

## 🔧 **Configuration**

### **Environment Variables**

| Variable | Default | Description |
|----------|---------|-------------|
| `PYTHONPATH` | `/app` | Python path |
| `RAY_DISABLE_IMPORT_WARNING` | `1` | Disable Ray warnings |
| `RAY_HEAD_SERVICE_PORT` | `6379` | Ray head service port |

### **Resource Allocation**

The Docker configuration includes:
- **Memory**: 4GB limit, 2GB reservation
- **CPU**: 2 cores limit, 1 core reservation
- **Workers**: 4 uvicorn workers for concurrent requests

## ⚡ **Async Architecture**

### **Concurrent Request Handling**

The API is designed to handle concurrent requests efficiently:

1. **FastAPI Async Endpoints**: All endpoints are `async` functions
2. **Ray Workers**: Separate Ray workers for each model type
3. **Uvicorn Workers**: Multiple uvicorn workers (4) for load distribution
4. **Concurrent Ensemble**: Uses `asyncio.gather()` for parallel model predictions

### **Request Flow**

```
Client Request → Uvicorn Worker → FastAPI Endpoint → Ray Worker → Model Prediction → Response
```

### **Ray Worker Architecture**

```python
@ray.remote
class PredictionWorker:
    def __init__(self, model_path: str, model_type: str):
        # Load model once per worker
        self.inference = load_model(model_path, model_type)
    
    def predict_single(self, image_path: str):
        # Predict using loaded model
        return self.inference.predict_single(image_path)
```

## 📊 **Performance Optimization**

### **Concurrent Request Support**

- **4 Uvicorn Workers**: Handle multiple requests simultaneously
- **Ray Workers**: Separate processes for each model type
- **Async I/O**: Non-blocking file operations and predictions
- **Model Caching**: Models loaded once per Ray worker

### **Expected Performance**

| Metric | Value |
|--------|-------|
| **Concurrent Requests** | 50+ simultaneous |
| **Response Time** | 100-500ms per request |
| **Throughput** | 100+ requests/second |
| **Memory Usage** | 2-4GB per container |

## 🔍 **Monitoring**

### **Health Check**

```bash
# Check API health
curl http://localhost:8001/health

# Expected response
{
  "status": "ok",
  "ray_initialized": true,
  "ray_workers_available": 2,
  "timestamp": "2024-01-01T12:00:00"
}
```

### **Logs**

```bash
# View container logs
docker logs cataract-api

# Follow logs in real-time
docker logs -f cataract-api
```

## 🧪 **Testing**

### **Load Testing**

```bash
# Test with curl
curl -X POST "http://localhost:8001/predict_clip" \
  -F "file=@test_image.jpg"

# Test ensemble prediction
curl -X POST "http://localhost:8001/predict_ensemble" \
  -F "file=@test_image.jpg"
```

### **Concurrent Testing**

```bash
# Test with multiple concurrent requests
for i in {1..10}; do
  curl -X POST "http://localhost:8001/predict_clip" \
    -F "file=@test_image.jpg" &
done
wait
```

## 🚨 **Troubleshooting**

### **Common Issues**

1. **Ray Initialization Failed**
   ```bash
   # Check Ray logs
   docker logs cataract-api | grep -i ray
   ```

2. **Model Loading Failed**
   ```bash
   # Ensure models are in the correct location
   ls -la models/
   ```

3. **Memory Issues**
   ```bash
   # Increase memory limit
   docker run --memory=8g cataract-api
   ```

### **Debug Mode**

```bash
# Run with debug logging
docker run -e LOG_LEVEL=DEBUG cataract-api
```

## 📈 **Scaling**

### **Horizontal Scaling**

```bash
# Scale to multiple instances
docker-compose up --scale cataract-api=3
```

### **Load Balancer**

```yaml
# nginx.conf
upstream cataract_api {
    server localhost:8001;
    server localhost:8002;
    server localhost:8003;
}
```

## 🔒 **Security**

### **Production Considerations**

1. **HTTPS**: Use reverse proxy with SSL
2. **Authentication**: Add API key authentication
3. **Rate Limiting**: Implement request rate limiting
4. **Input Validation**: Validate all input files
5. **Resource Limits**: Set appropriate memory/CPU limits

### **Security Headers**

```python
# Add to FastAPI app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)
```

## 📝 **API Endpoints**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict_clip` | POST | CLIP model prediction |
| `/predict_lgbm` | POST | LGBM model prediction |
| `/predict_ensemble` | POST | Ensemble prediction |
| `/` | GET | API information |

## 🎯 **Best Practices**

1. **Use Docker Compose** for easier management
2. **Monitor resource usage** with `docker stats`
3. **Set appropriate limits** for production
4. **Use volume mounts** for persistent data
5. **Implement health checks** for monitoring
6. **Use async endpoints** for better performance
7. **Cache models** in Ray workers
8. **Handle errors gracefully** with proper HTTP status codes

## 📞 **Support**

For issues or questions:
1. Check the logs: `docker logs cataract-api`
2. Verify model files are present
3. Ensure sufficient system resources
4. Check network connectivity
5. Review the API documentation at `/docs` 