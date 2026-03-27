from fastapi import FastAPI, File, UploadFile
import onnxruntime as ort
import numpy as np
from PIL import Image
import json
import io

app = FastAPI()

# 🔥 Load model once (VERY IMPORTANT)
session = ort.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name

# Load labels
with open("labels.json") as f:
    labels = json.load(f)

# Preprocessing function
def preprocess(image: Image.Image):
    image = image.resize((224, 224))
    img_array = np.array(image).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img_array = (img_array - mean) / std

    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)

    return img_array.astype(np.float32)

@app.get("/")
def home():
    return {"message": "Food Recognition API running 🚀"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Preprocess
        input_tensor = preprocess(image)

        # Inference
        outputs = session.run(None, {input_name: input_tensor})
        logits = outputs[0]

        # Prediction
        predicted_class = int(np.argmax(logits))
        confidence = float(np.max(logits))

        label = labels[str(predicted_class)]

        return {
            "prediction": label,
            "confidence": confidence
        }

    except Exception as e:
        return {"error": str(e)}