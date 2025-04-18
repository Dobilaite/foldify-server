from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import os
from ultralytics import YOLO

app = FastAPI()

# Allow requests from your Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, set to specific domain(s)
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


# This is required for Render to run the app on the correct host and port
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))  # Render provides PORT via env variable
    uvicorn.run("main:app", host="0.0.0.0", port=port)
