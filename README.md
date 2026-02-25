## object-detector-fastapi

### Run
docker-compose up --build

### Health
http://localhost:7001/health

### Predict
POST http://localhost:7001/predict (form-data: file=<image>)
