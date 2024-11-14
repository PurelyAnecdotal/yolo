from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

app = FastAPI(title="YOLO Object Detection API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def index():
    return "200"

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    """
    Endpoint to perform object detection on uploaded images
    """
    try:
        # Verify that model is loaded
        if model is None:
            return JSONResponse(
                status_code=500,
                content={"error": "Model not properly loaded"}
            )
            
        # Read and process the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Perform detection
        results = model(image)
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Get confidence score
                confidence = float(box.conf[0])
                
                # Get class name
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "class": class_name
                })
        
        return {
            "success": True,
            "detections": detections,
            "message": f"Processed {file.filename} successfully",
            "num_detections": len(detections)
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )

@app.get("/health")
async def health_check():
    """
    Endpoint to check if the server is running and model is loaded
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }
