from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
import joblib
import numpy as np
from train_model import decode_prediction  # ✅ Import the decoder function

app = FastAPI()

@app.get("/web")
def serve_web():
    return FileResponse("index.html")

# Load model
model = joblib.load("model.joblib")

# Request schema
class InputFeatures(BaseModel):
    features: list[float]

# Root route
@app.get("/")
def read_root():
    return {"message": "Welcome to ML API"}

# Predict route
@app.post("/predict")
def predict(data: InputFeatures):
    features = np.array(data.features).reshape(1, -1)
    prediction = model.predict(features)
    category = decode_prediction(prediction)
    
    return {
        "prediction": {
            "category": category,
            "number": int(prediction[0])
        }
    }
