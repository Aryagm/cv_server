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
    """Detect sidewalk boundaries using a point-based approach for more natural curves."""
    height, width = image.shape[:2]
    output = image.copy()
    center_x = width / 2
    
    # Store current position for movement detection
    current_position = center_x
    position_history.append(current_position)
    
    # Process the image to find edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Smaller kernel for finer details
    edges = cv2.Canny(blur, 30, 90)  # Slightly higher thresholds for stronger edges
    
    # Create mask to focus on sidewalk region
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (int(width * 0.05), height),  # Wider at bottom
        (int(width * 0.95), height),
        (int(width * 0.65), int(height * 0.55)),  # Higher vanishing point
        (int(width * 0.35), int(height * 0.55))
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Debug visualization for edge detection
    debug_image = np.zeros((height, width, 3), dtype=np.uint8)
    debug_image[masked_edges > 0] = [0, 255, 255]  # Yellow for edges
    
    # Line detection with improved parameters
    lines = cv2.HoughLinesP(
        masked_edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=25,  # Lower threshold to detect more lines
        minLineLength=40,
        maxLineGap=50
    )
    
    # Collect edge points from detected lines
    left_points = []
    right_points = []
    threshold_y = int(height * 0.7)  # Look higher up in the image
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Skip nearly horizontal lines
            if abs(y2 - y1) < 10:
                continue
                
            # Calculate slope and intercept for the line
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            
            # Only use lines with reasonable slope
            if 0.2 < abs(slope) < 10:
                # Draw detected lines for debugging
                cv2.line(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Project line to bottom of image
                if y1 != y2:  # Avoid division by zero
                    bottom_x = int(x1 + (height - y1) * (x2 - x1) / (y2 - y1))
                    if 0 <= bottom_x < width:  # Ensure point is in frame
                        if bottom_x < center_x:
                            left_points.append(bottom_x)
                        else:
                            right_points.append(bottom_x)
                
                # Also add original line points if they're low enough in the frame
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
    
    # Calculate confidence based on number and consistency of detected points
    left_confidence = 0
    right_confidence = 0
    left_boundary = int(width * 0.3)  # Default values
    right_boundary = int(width * 0.7)
    
    if left_points:
        left_std = np.std(left_points) if len(left_points) > 1 else width
        left_confidence = min(1.0, len(left_points) / 10.0) * max(0.0, 1.0 - left_std / (width * 0.3))
        # Use more robust estimator - trim outliers before taking median
        sorted_left = np.sort(left_points)
        trim_count = max(0, int(len(sorted_left) * 0.2))  # Trim 20% from each end
        if len(sorted_left) > trim_count * 2 + 1:
            trimmed_left = sorted_left[trim_count:-trim_count]
            left_boundary = int(np.median(trimmed_left))
        else:
            left_boundary = int(np.median(left_points))
    
    if right_points:
        right_std = np.std(right_points) if len(right_points) > 1 else width
        right_confidence = min(1.0, len(right_points) / 10.0) * max(0.0, 1.0 - right_std / (width * 0.3))
        # Trim outliers for right side too
        sorted_right = np.sort(right_points)
        trim_count = max(0, int(len(sorted_right) * 0.2))
        if len(sorted_right) > trim_count * 2 + 1:
            trimmed_right = sorted_right[trim_count:-trim_count]
            right_boundary = int(np.median(trimmed_right))
        else:
            right_boundary = int(np.median(right_points))
    
    # Ensure left is to the left of right
    if left_boundary > right_boundary:
        left_boundary, right_boundary = right_boundary, left_boundary
    
    # Apply constraints to prevent unreasonable sidewalk widths
    sidewalk_width = right_boundary - left_boundary
    if sidewalk_width < width * 0.2:  # Too narrow
        center = (left_boundary + right_boundary) // 2
        left_boundary = max(0, center - int(width * 0.15))
        right_boundary = min(width, center + int(width * 0.15))
    elif sidewalk_width > width * 0.6:  # Too wide
        center = (left_boundary + right_boundary) // 2
        left_boundary = max(0, center - int(width * 0.25))
        right_boundary = min(width, center + int(width * 0.25))
    
    # Add to boundary history for smoothing
    boundary_history.append((left_boundary, right_boundary))
    
    # Use a weighted average of recent boundaries for smoothness
    if len(boundary_history) > 1:
        # More weight to recent frames, adapted to history length
        max_weights = [0.1, 0.15, 0.2, 0.25, 0.3]
        weights = max_weights[-len(boundary_history):]
        weights = [w/sum(weights) for w in weights]
        
        # Calculate weighted average
        left_avg = int(sum(b[0] * w for b, w in zip(boundary_history, reversed(weights))))
        right_avg = int(sum(b[1] * w for b, w in zip(boundary_history, reversed(weights))))
        
        # Only use the smoothed values if they're not too different from current detection
        if abs(left_avg - left_boundary) < width * 0.1 and abs(right_avg - right_boundary) < width * 0.1:
            left_boundary, right_boundary = left_avg, right_avg
    
    # Create sidewalk visualization
    overlay = output.copy()
    
    # Define sidewalk shape with more natural curve
    vanishing_y = int(height * 0.55)  # Vanishing point height
    vanishing_x_left = int(center_x - width * 0.1)  # Vanishing point shifts slightly left
    vanishing_x_right = int(center_x + width * 0.1)
    
    # Create curved sidewalk path using bezier-like points
    sidewalk_left_x = [left_boundary]
    sidewalk_right_x = [right_boundary]
    
    # Generate points along the curve at different y positions
    for i in range(1, 5):
        ratio = i / 4.0
        y_pos = height - ratio * (height - vanishing_y)
        
        # Calculate curve points with more natural bending
        # Left side curves more toward center as we go up
        left_x = int(left_boundary * (1.0 - ratio) + vanishing_x_left * ratio)
        
        # Right side curves more toward center as we go up
        right_x = int(right_boundary * (1.0 - ratio) + vanishing_x_right * ratio)
        
        sidewalk_left_x.append(left_x)
        sidewalk_right_x.append(right_x)
    
    # Create sidewalk polygon points
    sidewalk_points = []
    for i in range(len(sidewalk_left_x)):
        y_pos = height - i * (height - vanishing_y) / 4.0
        sidewalk_points.append([sidewalk_left_x[i], int(y_pos)])
    
    for i in range(len(sidewalk_right_x) - 1, -1, -1):
        y_pos = height - i * (height - vanishing_y) / 4.0
        sidewalk_points.append([sidewalk_right_x[i], int(y_pos)])
    
    sidewalk_poly = np.array([sidewalk_points], dtype=np.int32)
    
    # Draw sidewalk with more natural coloring
    cv2.fillPoly(overlay, sidewalk_poly, (0, 150, 0))  # Darker green for better visibility
    
    # Add distance markers on the sidewalk with perspective
    for i in range(1, 6):
        y_pos = height - int((height - vanishing_y) * i / 5)
        ratio = i / 5.0
        left_idx = min(int(ratio * (len(sidewalk_left_x) - 1)), len(sidewalk_left_x) - 1)
        right_idx = min(int(ratio * (len(sidewalk_right_x) - 1)), len(sidewalk_right_x) - 1)
        left_x = sidewalk_left_x[left_idx]
        right_x = sidewalk_right_x[right_idx]
        cv2.line(overlay, (left_x, y_pos), (right_x, y_pos), (255, 255, 255), max(1, 3 - i//2))
    
    # Add the overlay with transparency
    cv2.addWeighted(overlay, 0.4, output, 0.6, 0, output)
    
    # Draw the sidewalk boundaries with a more appealing style
    for i in range(len(sidewalk_left_x) - 1):
        y1 = int(height - i * (height - vanishing_y) / 4.0)
        y2 = int(height - (i+1) * (height - vanishing_y) / 4.0)
        cv2.line(output, (sidewalk_left_x[i], y1), (sidewalk_left_x[i+1], y2), (0, 0, 255), 3)
        cv2.line(output, (sidewalk_right_x[i], y1), (sidewalk_right_x[i+1], y2), (0, 0, 255), 3)
    
    # Add debug overlay if needed (uncomment to show edge detection)
    # cv2.addWeighted(debug_image, 0.3, output, 0.7, 0, output)
    
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