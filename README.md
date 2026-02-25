HEAD
## object-detector-fastapi

### Run
docker-compose up --build

### Health
http://localhost:7001/health

### Predict
POST http://localhost:7001/predict (form-data: file=<image>)

# object-detector-fastapi
FastAPI + Docker service for image classification inference (PyTorch model).
d78eebcd311d56b6856e5169dc423e24837a0368
