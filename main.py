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

position_history = deque(maxlen=10)  # Track user position for stability


def detect_sidewalk_boundaries(image):
    """Detect sidewalk boundaries using edge detection and Hough Transform with boundary tracking."""
    height, width = image.shape[:2]
    output = image.copy()
    center_x = width / 2
    
    # Store current position for movement detection
    current_position = center_x
    position_history.append(current_position)
    
    # Process the image to find edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 20, 80)
    
    # Create mask to focus on sidewalk region
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (int(width * 0.1), height),
        (int(width * 0.9), height),
        (int(width * 0.6), int(height * 0.6)),
        (int(width * 0.4), int(height * 0.6))
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Line detection with HoughLinesP
    lines = cv2.HoughLinesP(
        masked_edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=30,
        minLineLength=50,  # Longer lines for better stability
        maxLineGap=100
    )
    
    # Debug image to visualize detected lines
    line_image = np.zeros_like(image)
    
    # Collect points for left and right boundaries
    left_points = []
    right_points = []
    threshold_y = int(height * 0.8)  # Focus on bottom 20% of image for stability
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Skip nearly horizontal lines
            if abs(y2 - y1) < 5:
                continue
            
            # Calculate slope    
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            
            # Only use lines with reasonable slope
            if abs(slope) > 0.3:
                # Draw detected lines in debug image
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Collect points from bottom part of image
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
    
    # Calculate boundaries based on detected points
    left_boundary = None
    right_boundary = None
    
    # Use median for robustness against outliers
    if left_points:
        left_std = np.std(left_points) if len(left_points) > 1 else width
        # Filter out outliers if we have enough points
        if len(left_points) > 3 and left_std < width * 0.2:
            sorted_left = np.sort(left_points)
            q1 = np.percentile(sorted_left, 25)
            q3 = np.percentile(sorted_left, 75)
            iqr = q3 - q1
            left_filtered = [p for p in left_points if q1 - 1.5*iqr <= p <= q3 + 1.5*iqr]
            if left_filtered:
                left_boundary = int(np.median(left_filtered))
            else:
                left_boundary = int(np.median(left_points))
        else:
            left_boundary = int(np.median(left_points))
    
    if right_points:
        right_std = np.std(right_points) if len(right_points) > 1 else width
        # Filter out outliers if we have enough points
        if len(right_points) > 3 and right_std < width * 0.2:
            sorted_right = np.sort(right_points)
            q1 = np.percentile(sorted_right, 25)
            q3 = np.percentile(sorted_right, 75)
            iqr = q3 - q1
            right_filtered = [p for p in right_points if q1 - 1.5*iqr <= p <= q3 + 1.5*iqr]
            if right_filtered:
                right_boundary = int(np.median(right_filtered))
            else:
                right_boundary = int(np.median(right_points))
        else:
            right_boundary = int(np.median(right_points))
    
    # Apply default boundaries if detection failed
    if left_boundary is None:
        left_boundary = int(width * 0.3)
    if right_boundary is None:
        right_boundary = int(width * 0.7)
    
    # Ensure left is to the left of right
    if left_boundary > right_boundary:
        left_boundary, right_boundary = right_boundary, left_boundary
    
    # Add to boundary history for smoothing
    boundary_history.append((left_boundary, right_boundary))
    
    # Apply temporal smoothing with weighted average
    if len(boundary_history) > 1:
        weights = [0.1, 0.15, 0.25, 0.5][:len(boundary_history)]
        weights = [w/sum(weights) for w in weights]
        
        # Calculate weighted average
        left_avg = int(sum(b[0] * w for b, w in zip(boundary_history, reversed(weights))))
        right_avg = int(sum(b[1] * w for b, w in zip(boundary_history, reversed(weights))))
        
        # Use smoothed values
        left_boundary = left_avg
        right_boundary = right_avg
    
    # Enforce reasonable sidewalk width constraints
    sidewalk_width = right_boundary - left_boundary
    if sidewalk_width < width * 0.25:  # Too narrow
        center = (left_boundary + right_boundary) // 2
        left_boundary = center - int(width * 0.15)
        right_boundary = center + int(width * 0.15)
    elif sidewalk_width > width * 0.75:  # Too wide
        center = (left_boundary + right_boundary) // 2
        left_boundary = center - int(width * 0.3)
        right_boundary = center + int(width * 0.3)
    
    # Define the vanishing point for perspective
    vanishing_y = int(height * 0.6)
    vanishing_x = int(width * 0.5)
    
    # Calculate the top points (converging toward vanishing point)
    left_top_x = int(left_boundary * 0.8 + vanishing_x * 0.2)
    right_top_x = int(right_boundary * 0.8 + vanishing_x * 0.2)
    
    # Create sidewalk visualization
    overlay = output.copy()
    
    # Create a polygon for the detected sidewalk region
    sidewalk_poly = np.array([
        [(left_boundary, height), 
         (left_top_x, vanishing_y),
         (right_top_x, vanishing_y),
         (right_boundary, height)]
    ], dtype=np.int32)
    
    # Draw filled sidewalk area with semi-transparent green
    cv2.fillPoly(overlay, sidewalk_poly, (0, 180, 0))
    
    # Add distance markers with perspective effect
    for i in range(1, 6):
        # Calculate y-position with perspective effect
        y_pos = height - int((height - vanishing_y) * i / 5)
        
        # Calculate x-positions with perspective effect
        ratio = i / 5.0
        left_x = int(left_boundary * (1.0 - ratio) + left_top_x * ratio)
        right_x = int(right_boundary * (1.0 - ratio) + right_top_x * ratio)
        
        # Draw perspective-correct distance marker
        cv2.line(overlay, (left_x, y_pos), (right_x, y_pos), 
                 (255, 255, 255), max(1, 3 - i//2))
    
    # Add the overlay with transparency
    cv2.addWeighted(overlay, 0.4, output, 0.6, 0, output)
    
    # Draw sidewalk boundary lines with attractive style
    cv2.line(output, (left_boundary, height), (left_top_x, vanishing_y), (0, 0, 255), 3)
    cv2.line(output, (right_boundary, height), (right_top_x, vanishing_y), (0, 0, 255), 3)
    
    # Add detected line debug overlay
    cv2.addWeighted(line_image, 0.3, output, 1.0, 0, output)
    
    return output, (left_boundary, right_boundary)


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