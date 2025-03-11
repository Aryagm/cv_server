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

# Simple boundary smoothing with a short history
boundary_history = deque(maxlen=5)

last_alert_times = {}
ALERT_COOLDOWN = 5  # seconds

def add_alert(alerts, alert):
    """Add an alert only if the last same alert was sent more than ALERT_COOLDOWN seconds ago."""
    current_time = time.time()
    last_time = last_alert_times.get(alert, 0)
    
    if current_time - last_time >= ALERT_COOLDOWN:
        last_alert_times[alert] = current_time
        alerts.append(alert)

# Define a Pydantic model for incoming frame data
class FrameData(BaseModel):
    image: str

position_history = deque(maxlen=10)  # Track user position for stability

def detect_sidewalk_boundaries(image):
    """Detect sidewalk boundaries with improved side-view support."""
    height, width = image.shape[:2]
    center_x = width / 2
    
    # Store current position for movement detection
    current_position = center_x
    position_history.append(current_position)
    
    # Process the image to find edges with optimized parameters
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 20, 80)  # Lower thresholds to detect more subtle edges
    
    # Create wider mask to capture sidewalk even when off-center
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, height),  # Full width at bottom
        (width, height),
        (int(width * 0.7), int(height * 0.6)),  # Asymmetric top to handle side views
        (int(width * 0.3), int(height * 0.6))
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Line detection with parameters optimized for side views
    lines = cv2.HoughLinesP(
        masked_edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=30,  # Lower threshold to detect more lines
        minLineLength=50,
        maxLineGap=100
    )
    
    # Create debug image for visualization
    line_image = np.zeros_like(image)
    
    # Initialize collections for line segments
    left_candidates = []  # Will store (x, slope) pairs
    right_candidates = []  # Will store (x, slope) pairs
    
    # Process detected lines
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Skip nearly horizontal lines
            if abs(y2 - y1) < 10:
                continue
                
            # Calculate slope
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            
            # Only use lines with reasonable slope for sidewalk boundaries
            if 0.3 < abs(slope) < 5:
                # Project line to bottom of image for better classification
                if y1 != y2:  # Avoid division by zero
                    x_bottom = int(x1 + (height - y1) * (x2 - x1) / (y2 - y1))
                    
                    # Classify as left/right based on slope direction, not just position
                    if slope > 0:  # Rising from left to right -> likely a left boundary
                        left_candidates.append((x_bottom, slope))
                        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    elif slope < 0:  # Falling from left to right -> likely a right boundary
                        right_candidates.append((x_bottom, slope))
                        cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
    
    # Find most consistent boundary positions using clustering
    left_boundary = None
    right_boundary = None
    
    # Process left candidates
    if left_candidates:
        # Sort by x position
        left_candidates.sort()
        
        # Check if we have clusters of lines
        if len(left_candidates) >= 3:
            # Extract x positions
            left_x = [p[0] for p in left_candidates]
            
            # Use DBSCAN clustering to find the most consistent group
            from sklearn.cluster import DBSCAN
            try:
                # Simple clustering - can be replaced with manual approach if needed
                clustering = DBSCAN(eps=width*0.1, min_samples=2).fit(np.array(left_x).reshape(-1, 1))
                if len(set(clustering.labels_)) > 1:  # If we found clusters
                    # Find the largest cluster
                    largest_cluster = max(set(clustering.labels_), key=list(clustering.labels_).count)
                    if largest_cluster != -1:  # -1 is noise
                        cluster_points = [left_x[i] for i, label in enumerate(clustering.labels_) if label == largest_cluster]
                        left_boundary = int(np.median(cluster_points))
                    else:
                        left_boundary = int(np.median(left_x))
                else:
                    left_boundary = int(np.median(left_x))
            except:
                # Fallback if clustering fails
                left_boundary = int(np.median(left_x))
        else:
            # Just take the median if few points
            left_boundary = int(np.median([p[0] for p in left_candidates]))
    
    # Process right candidates
    if right_candidates:
        # Sort by x position
        right_candidates.sort()
        
        # Check if we have clusters of lines
        if len(right_candidates) >= 3:
            # Extract x positions
            right_x = [p[0] for p in right_candidates]
            
            # Use DBSCAN clustering to find the most consistent group
            from sklearn.cluster import DBSCAN
            try:
                # Simple clustering - can be replaced with manual approach if needed
                clustering = DBSCAN(eps=width*0.1, min_samples=2).fit(np.array(right_x).reshape(-1, 1))
                if len(set(clustering.labels_)) > 1:  # If we found clusters
                    # Find the largest cluster
                    largest_cluster = max(set(clustering.labels_), key=list(clustering.labels_).count)
                    if largest_cluster != -1:  # -1 is noise
                        cluster_points = [right_x[i] for i, label in enumerate(clustering.labels_) if label == largest_cluster]
                        right_boundary = int(np.median(cluster_points))
                    else:
                        right_boundary = int(np.median(right_x))
                else:
                    right_boundary = int(np.median(right_x))
            except:
                # Fallback if clustering fails
                right_boundary = int(np.median(right_x))
        else:
            # Just take the median if few points
            right_boundary = int(np.median([p[0] for p in right_candidates]))
    
    # Handle cases where only one boundary is detected
    if left_boundary is not None and right_boundary is None:
        # Estimate right boundary based on typical sidewalk width
        right_boundary = min(width - 20, left_boundary + int(width * 0.3))
    elif left_boundary is None and right_boundary is not None:
        # Estimate left boundary based on typical sidewalk width
        left_boundary = max(20, right_boundary - int(width * 0.3))
    elif left_boundary is None and right_boundary is None:
        # No boundaries detected, use default positions
        left_boundary = int(width * 0.35)
        right_boundary = int(width * 0.65)
    
    # Ensure boundaries are within image
    left_boundary = max(0, min(left_boundary, width-1))
    right_boundary = max(0, min(right_boundary, width-1))
    
    # Ensure left is to the left of right
    if left_boundary > right_boundary:
        left_boundary, right_boundary = right_boundary, left_boundary
    
    # Handle unreasonably narrow or wide sidewalks
    sidewalk_width = right_boundary - left_boundary
    if sidewalk_width < width * 0.15:  # Too narrow
        midpoint = (left_boundary + right_boundary) // 2
        left_boundary = max(0, midpoint - int(width * 0.1))
        right_boundary = min(width, midpoint + int(width * 0.1))
    elif sidewalk_width > width * 0.6:  # Too wide
        midpoint = (left_boundary + right_boundary) // 2
        left_boundary = max(0, midpoint - int(width * 0.25))
        right_boundary = min(width, midpoint + int(width * 0.25))
    
    # Add to boundary history for temporal smoothing
    boundary_history.append((left_boundary, right_boundary))
    
    # Apply temporal smoothing for stability
    if len(boundary_history) > 2:
        # More weight to recent frames
        weights = [0.2, 0.3, 0.5][:len(boundary_history)]
        weights = [w/sum(weights) for w in weights]
        
        # Calculate weighted average
        left_avg = int(sum(b[0] * w for b, w in zip(boundary_history, reversed(weights))))
        right_avg = int(sum(b[1] * w for b, w in zip(boundary_history, reversed(weights))))
        
        # Only use smoothed values if they're not drastically different
        if abs(left_avg - left_boundary) < width * 0.1:
            left_boundary = left_avg
        if abs(right_avg - right_boundary) < width * 0.1:
            right_boundary = right_avg
    
    # Define visualization properties
    vanishing_y = int(height * 0.6)
    
    # Calculate vanishing point (should be between boundaries, weighted toward center)
    vanishing_x = int((left_boundary + right_boundary) / 2 * 0.7 + width / 2 * 0.3)
    vanishing_x = max(min(vanishing_x, width-20), 20)  # Keep within image
    
    # Create a more robust sidewalk visualization
    output = image.copy()
    overlay = output.copy()
    
    # Create the sidewalk polygon with perspective
    sidewalk_points = np.array([
        [left_boundary, height],
        [int(left_boundary * 0.7 + vanishing_x * 0.3), vanishing_y],
        [int(right_boundary * 0.7 + vanishing_x * 0.3), vanishing_y],
        [right_boundary, height]
    ], np.int32)
    
    # Draw filled sidewalk with green overlay
    cv2.fillPoly(overlay, [sidewalk_points], (0, 150, 0))
    
    # Add distance markers with perspective effect
    for i in range(1, 6):
        # Calculate positions with perspective
        y_pos = height - int((height - vanishing_y) * i / 5)
        ratio = i / 5.0
        left_x = int(left_boundary * (1.0 - ratio) + (left_boundary * 0.7 + vanishing_x * 0.3) * ratio)
        right_x = int(right_boundary * (1.0 - ratio) + (right_boundary * 0.7 + vanishing_x * 0.3) * ratio)
        
        # Draw the distance marker line
        cv2.line(overlay, (left_x, y_pos), (right_x, y_pos), 
                 (255, 255, 255), max(1, 3 - i//2))
    
    # Add overlay with transparency
    cv2.addWeighted(overlay, 0.4, output, 0.6, 0, output)
    
    # Draw boundary lines
    cv2.line(output, 
             (left_boundary, height), 
             (int(left_boundary * 0.7 + vanishing_x * 0.3), vanishing_y), 
             (0, 0, 255), 3)
    cv2.line(output, 
             (right_boundary, height), 
             (int(right_boundary * 0.7 + vanishing_x * 0.3), vanishing_y), 
             (0, 0, 255), 3)
    
    # Add debug overlay of detected lines
    cv2.addWeighted(line_image, 0.3, output, 1.0, 0, output)
    
    # Add boundaries
    boundaries = (left_boundary, right_boundary)
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
                add_alert(alerts, f"{label} ahead.")
            elif label_lower in ["traffic sign", "stop sign"]:
                add_alert(alerts, "Sign detected.")
            elif label_lower == "crosswalk":
                add_alert(alerts, "Crosswalk ahead.")

    BOUNDARY_WARNING_THRESHOLD = 0.30  # Warn when within 20% of sidewalk width from edge
    
# Modify the user position indicator and guidance section in process_frame
    # Add user position indicator and sidewalk guidance
    if boundaries:
        left_boundary, right_boundary = boundaries
        height, width, _ = img.shape
        user_point = (width // 2, height - 20)
        
        # Calculate position within sidewalk
        sidewalk_width = right_boundary - left_boundary
        position_percent = ((user_point[0] - left_boundary) / max(1, sidewalk_width)) * 100
        position_percent = max(0, min(100, position_percent))  # Clamp to 0-100%
        
        # Determine if user is off-course or approaching boundary
        off_sidewalk = user_point[0] < left_boundary or user_point[0] > right_boundary
        
        # Calculate distance from boundaries as percentage of sidewalk width
        distance_from_left = (user_point[0] - left_boundary) / max(1, sidewalk_width)
        distance_from_right = (right_boundary - user_point[0]) / max(1, sidewalk_width)
        
        # Check if user is in warning zone (close to boundary)
        near_left_boundary = 0 <= distance_from_left < BOUNDARY_WARNING_THRESHOLD
        near_right_boundary = 0 <= distance_from_right < BOUNDARY_WARNING_THRESHOLD
        
        # Determine indicator color based on position
        if off_sidewalk:
            indicator_color = (0, 0, 255)  # Red for off sidewalk
        elif near_left_boundary or near_right_boundary:
            indicator_color = (0, 165, 255)  # Orange for near boundary
        else:
            indicator_color = (0, 255, 255)  # Yellow for safe
        
        # Draw an attractive user position indicator (triangle pointing up)
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
        
        #Add directional guidance based on position
        if off_sidewalk:
            # Critical alert for off sidewalk
            add_alert(alerts, "WARNING: You might be going off the sidewalk.")
            
            # if user_point[0] < left_boundary:
            #     cv2.putText(processed_img, "MOVE RIGHT", (width//2 - 80, height - 80),
            #                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            # else:
            #     cv2.putText(processed_img, "MOVE LEFT", (width//2 - 80, height - 80),
            #                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        if near_left_boundary:
            # Warning alert for approaching left boundary
            add_alert(alerts, "WARNING: You might be going off the sidewalk.")
            cv2.putText(processed_img, "DRIFT", (width//2 - 80, height - 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
        elif near_right_boundary:
            # Warning alert for approaching right boundary
            add_alert(alerts, "WARNING: You might be going off the sidewalk.")
            cv2.putText(processed_img, "DRIFT ", (width//2 - 80, height - 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)

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