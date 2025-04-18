from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
from ultralytics import YOLO

app = FastAPI()

# Allow requests from your Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, set to your domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
model = YOLO("best.pt")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    results = model(image)

    if len(results[0].boxes) == 0:
        return JSONResponse(content={"category": "none"}, status_code=200)

    class_id = int(results[0].boxes.cls[0])
    category = model.names[class_id]

    return {"category": category}
