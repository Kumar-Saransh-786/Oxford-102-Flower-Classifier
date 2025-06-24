# app.py

import io
import numpy as np
import cv2
import requests
import os

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import uvicorn

app = FastAPI(title="Oxford-102 Flower Classifier")

# ——— 1) Locate your model file —————————————————————————————————————
# This grabs the folder where app.py lives:
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Change this name to whichever file you want to load:
MODEL_NAME = "flowers102_model.keras"   # or "flower_classifier.h5"

# Build the full path to that file:
MODEL_PATH = os.path.join(BASE_DIR, MODEL_NAME)

# ——— 2) Load the model at startup ————————————————————————————————————
MODEL = load_model(MODEL_PATH)

GIST_URL = (
    "https://gist.githubusercontent.com/"
    "JosephKJ/94c7728ed1a8e0cd87fe6a029769cde1"
    "/raw/Oxford-102_Flower_dataset_labels.txt"
)
resp = requests.get(GIST_URL)
FLOWER_NAMES = [
    line.strip().strip("'\"")
    for line in resp.text.splitlines()
    if line.strip()
]
if len(FLOWER_NAMES) != 102:
    raise RuntimeError(f"Expected 102 names, got {len(FLOWER_NAMES)}")

IMG_SIZE = (224, 224)

# ——— 2) Prediction endpoint ————————————————————————————————————————
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 2a) Read & decode image bytes
    data = await file.read()
    np_arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # 2b) Preprocess
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = preprocess_input(img.astype("float32"))
    img = np.expand_dims(img, axis=0)

    # 2c) Model inference
    preds = MODEL.predict(img)[0]
    idx   = int(np.argmax(preds))
    conf  = float(preds[idx])

    return JSONResponse({
        "predicted_class_index": idx,
        "flower_name": FLOWER_NAMES[idx],
        "confidence": round(conf, 4)
    })

# ——— 3) Run the app —————————————————————————————————————————————
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
