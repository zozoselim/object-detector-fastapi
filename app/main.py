import io
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from app.model import InferenceService

app = FastAPI(title="ML Inference API", version="1.0.0")

MODEL_PATH = "models/detector.pth"
LABEL_MAP_PATH = "models/label_map.pkl"

service = InferenceService(MODEL_PATH, LABEL_MAP_PATH, device="cpu")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    img = Image.open(io.BytesIO(content))
    label, probability = service.predict(img)
    return {"label": label, "probability": probability}
