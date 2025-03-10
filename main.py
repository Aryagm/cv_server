import base64
import cv2
import numpy as np
import time
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
from collections import deque

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
MODEL = YOLO("yolo11n.pt")

# Global dictionary to store the last timestamp for each alert
last_alert_times = {}
ALERT_COOLDOWN = 10  # seconds

# Simple boundary smoothing with a short history
boundary_history = deque(maxlen=5)

def add_alert(alerts, alert):
    """Add an alert only if the last same alert was sent more than ALERT_COOLDOWN seconds ago."""
    current_time = time.time()
    if current_time - last_alert_times.get(alert, 0) >= ALERT_COOLDOWN:
        last_alert_times[alert] = current_time
        alerts.append(alert)

# Define a Pydantic model for incoming frame data
class FrameData(BaseModel):
    image: str

def detect_sidewalk_boundaries(image):
    """Detect sidewalk boundaries using edge detection and Hough Transform."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 30, 100)
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (int(width * 0.1), height),
        (int(width * 0.9), height),
        (int(width * 0.6), int(height * 0.6)),
        (int(width * 0.4), int(height * 0.6))
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi / 180,
                            threshold=40, minLineLength=60, maxLineGap=80)
    line_image = np.copy(image) * 0
    left_points = []
    right_points = []
    center_x = width / 2
    threshold_y = int(height * 0.8)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            if abs(slope) > 0.3:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
                if y1 > threshold_y:
                    if x1 < center_x:
                        left_points.append(x1)
                    else:
                        right_points.append(x1)
                if y2 > threshold_y:
                    if x2 < center_x:
                        left_points.append(x2)
                    else:
                        right_points.append(x2)
    boundaries = None
    if left_points and right_points:
        left_boundary = int(np.median(left_points))
        right_boundary = int(np.median(right_points))
        boundaries = (left_boundary, right_boundary)
    output = cv2.addWeighted(image, 0.8, line_image, 1, 1)
    if boundaries is not None:
        left_boundary, right_boundary = boundaries
        cv2.line(output, (left_boundary, height), (left_boundary, int(height * 0.6)), (255, 0, 0), 2)
        cv2.line(output, (right_boundary, height), (right_boundary, int(height * 0.6)), (255, 0, 0), 2)
    return output, boundaries


@app.post("/process_frame/")
async def process_frame(frame_data: FrameData):
    image_data = frame_data.image
    if not image_data:
        raise HTTPException(status_code=400, detail="No image provided")
    # Remove the data URL prefix if present
    if "," in image_data:
        image_data = image_data.split(",")[1]

    try:
        img_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

    alerts = []

    # Run object detection and sidewalk detection
    yolo_results = MODEL.predict(img, device=device)
    processed_img, boundaries = detect_sidewalk_boundaries(img)

    # Draw YOLO detection results on the processed image
    for result in yolo_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = MODEL.names[cls] if hasattr(MODEL, "names") else str(cls)
            label_lower = label.lower()
            
            # Choose color based on object type
            color = (0, 255, 0)  # Default: green
            if label_lower == "person":
                color = (255, 0, 0)  # Blue
            elif label_lower in ["car", "truck", "bus", "motorcycle", "bicycle"]:
                color = (0, 0, 255)  # Red
            elif label_lower in ["traffic sign", "stop sign"]:
                color = (0, 255, 255)  # Yellow
            elif label_lower == "crosswalk":
                color = (255, 255, 0)  # Cyan
                
            # Draw the bounding box and label
            cv2.rectangle(processed_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(processed_img, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add relevant alerts
            if label_lower in ["person", "car", "truck", "bus", "motorcycle", "bicycle"]:
                add_alert(alerts, f"{label} ahead!")
            elif label_lower in ["traffic sign", "stop sign"]:
                add_alert(alerts, "Sign detected!")
            elif label_lower == "crosswalk":
                add_alert(alerts, "Crosswalk ahead!")

    # Add user position indicator and sidewalk guidance
    if boundaries:
        left_boundary, right_boundary = boundaries
        height, width, _ = img.shape
        user_point = (width // 2, height - 20)
        
        # Calculate position within sidewalk
        sidewalk_width = right_boundary - left_boundary
        position_percent = ((user_point[0] - left_boundary) / max(1, sidewalk_width)) * 100
        position_percent = max(0, min(100, position_percent))  # Clamp to 0-100%
        
        # Determine if user is off-course
        off_sidewalk = user_point[0] < left_boundary or user_point[0] > right_boundary
        
        # Draw an attractive user position indicator (triangle pointing up)
        indicator_color = (0, 0, 255) if off_sidewalk else (0, 255, 255)
        triangle_pts = np.array([
            [user_point[0], user_point[1] - 20],
            [user_point[0] - 10, user_point[1]],
            [user_point[0] + 10, user_point[1]]
        ], np.int32)
        cv2.fillPoly(processed_img, [triangle_pts], indicator_color)
        cv2.polylines(processed_img, [triangle_pts], True, (0, 0, 0), 2)
        
        # Add position indicator text with better styling
        cv2.rectangle(processed_img, (width//2 - 100, height - 60), (width//2 + 100, height - 30), (0, 0, 0), -1)
        cv2.rectangle(processed_img, (width//2 - 100, height - 60), (width//2 + 100, height - 30), (255, 255, 255), 1)
        position_text = f"Position: {position_percent:.1f}%"
        cv2.putText(processed_img, position_text, (width//2 - 80, height - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add directional guidance if off-course
        if off_sidewalk:
            add_alert(alerts, "WARNING: You are going off the sidewalk!")
            
            if user_point[0] < left_boundary:
                cv2.putText(processed_img, "MOVE RIGHT", (width//2 - 80, height - 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            else:
                cv2.putText(processed_img, "MOVE LEFT", (width//2 - 80, height - 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    # Encode the processed image to base64
    _, buffer = cv2.imencode('.jpg', processed_img)
    processed_img_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "alerts": alerts,
        "processed_image": "data:image/jpeg;base64," + processed_img_base64
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)