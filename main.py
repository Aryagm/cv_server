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
    """Detect sidewalk boundaries with enhanced curve detection capabilities."""
    height, width = image.shape[:2]
    output = image.copy()
    center_x = width / 2
    
    # Store current position for movement detection
    current_position = center_x
    position_history.append(current_position)
    
    # Process the image to find edges with parameters for better curve detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)  # Larger kernel for smoother curves
    edges = cv2.Canny(blur, 20, 80)  # Lower thresholds to detect more subtle curves
    
    # Create mask to focus on sidewalk region - narrower to focus on actual sidewalk
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (int(width * 0.1), height),  # Narrower at bottom
        (int(width * 0.9), height),
        (int(width * 0.6), int(height * 0.6)),  # Higher vanishing point
        (int(width * 0.4), int(height * 0.6))
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Debug visualization
    debug_image = np.zeros((height, width, 3), dtype=np.uint8)
    debug_image[masked_edges > 0] = [0, 255, 255]  # Yellow for edges
    
    # Use multiple y-levels to capture curve points at different heights
    y_levels = [int(height * (1 - i/10)) for i in range(5)]  # 5 levels from bottom to 50% height
    left_curve_points = []  # Will store (x,y) points for left curve
    right_curve_points = []  # Will store (x,y) points for right curve
    
    # Line detection with parameters optimized for curves
    lines = cv2.HoughLinesP(
        masked_edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=30,  # Higher threshold for more reliable lines
        minLineLength=30,  # Shorter to capture curve segments
        maxLineGap=100  # Larger gap to connect curve segments
    )
    
    # Process detected lines
    if lines is not None:
        # First, extract line segments and classify as left/right
        left_segments = []
        right_segments = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Skip nearly horizontal lines
            if abs(y2 - y1) < 15:
                continue
                
            # Calculate slope
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            
            # Only use lines with reasonable slope for sidewalk boundaries
            if 0.3 < abs(slope) < 8:
                # Classify as left or right based on position and slope
                if x1 < center_x or x2 < center_x:
                    if slope > 0:  # Positive slope for left boundary
                        left_segments.append(line[0])
                        cv2.line(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if x1 > center_x or x2 > center_x:
                    if slope < 0:  # Negative slope for right boundary
                        right_segments.append(line[0])
                        cv2.line(debug_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Sample points along each y-level from the line segments
        for y_level in y_levels:
            left_x_points = []
            right_x_points = []
            
            # Check intersections with left segments
            for x1, y1, x2, y2 in left_segments:
                # If line segment crosses this y-level
                if (y1 <= y_level <= y2) or (y2 <= y_level <= y1):
                    # Calculate x at this y-level using line equation
                    if y2 != y1:  # Avoid division by zero
                        x = int(x1 + (y_level - y1) * (x2 - x1) / (y2 - y1))
                        if 0 <= x < center_x:  # Ensure it's on the left side
                            left_x_points.append(x)
            
            # Check intersections with right segments
            for x1, y1, x2, y2 in right_segments:
                # If line segment crosses this y-level
                if (y1 <= y_level <= y2) or (y2 <= y_level <= y1):
                    # Calculate x at this y-level using line equation
                    if y2 != y1:  # Avoid division by zero
                        x = int(x1 + (y_level - y1) * (x2 - x1) / (y2 - y1))
                        if center_x <= x < width:  # Ensure it's on the right side
                            right_x_points.append(x)
            
            # If we found points at this y-level, add the median to our curve points
            if left_x_points:
                left_x = int(np.median(left_x_points))
                left_curve_points.append((left_x, y_level))
            
            if right_x_points:
                right_x = int(np.median(right_x_points))
                right_curve_points.append((right_x, y_level))
    
    # If we don't have enough curve points, add default points based on assumptions
    if len(left_curve_points) < 3:
        left_x = int(width * 0.35)  # Default left boundary
        left_curve_points = [(left_x, height)]
    
    if len(right_curve_points) < 3:
        right_x = int(width * 0.65)  # Default right boundary
        right_curve_points = [(right_x, height)]
    
    # Sort by y-coordinate (bottom to top)
    left_curve_points.sort(key=lambda p: -p[1])
    right_curve_points.sort(key=lambda p: -p[1])
    
    # Always ensure we have a point at the bottom
    if left_curve_points[0][1] != height:
        if left_curve_points:
            left_x_bottom = left_curve_points[0][0]
        else:
            left_x_bottom = int(width * 0.35)
        left_curve_points.insert(0, (left_x_bottom, height))
    
    if right_curve_points[0][1] != height:
        if right_curve_points:
            right_x_bottom = right_curve_points[0][0]
        else:
            right_x_bottom = int(width * 0.65)
        right_curve_points.insert(0, (right_x_bottom, height))
    
    # Get the bottom boundary points (where y=height)
    left_boundary = left_curve_points[0][0]
    right_boundary = right_curve_points[0][0]
    
    # Add to boundary history for smoothing - only bottom points
    boundary_history.append((left_boundary, right_boundary))
    
    # Apply smoothing to the boundary points over time
    if len(boundary_history) > 1:
        weights = [0.1, 0.2, 0.3, 0.4][:len(boundary_history)]
        weights = [w/sum(weights) for w in weights]
        
        # Calculate weighted average for bottom points
        left_avg = int(sum(b[0] * w for b, w in zip(boundary_history, reversed(weights))))
        right_avg = int(sum(b[1] * w for b, w in zip(boundary_history, reversed(weights))))
        
        # Update the bottom points with smoothed values
        if abs(left_avg - left_boundary) < width * 0.1:
            left_boundary = left_avg
            left_curve_points[0] = (left_avg, height)
        
        if abs(right_avg - right_boundary) < width * 0.1:
            right_boundary = right_avg
            right_curve_points[0] = (right_avg, height)
    
    # Adjust curve points to prevent unreasonable sidewalk width
    sidewalk_width = right_boundary - left_boundary
    if sidewalk_width < width * 0.2 or sidewalk_width > width * 0.6:
        center = (left_boundary + right_boundary) // 2
        left_boundary = max(0, center - int(width * 0.2))
        right_boundary = min(width, center + int(width * 0.2))
        
        # Update bottom points
        left_curve_points[0] = (left_boundary, height)
        right_curve_points[0] = (right_boundary, height)
    
    # Create a vanishing point for the top of the sidewalk
    vanishing_y = int(height * 0.6)
    vanishing_x = int(width * 0.5)
    
    # Ensure we have a top point that converges toward the vanishing point
    if len(left_curve_points) == 1 or left_curve_points[-1][1] > vanishing_y + 40:
        left_top_x = int(center_x - width * 0.1)
        left_curve_points.append((left_top_x, vanishing_y))
    
    if len(right_curve_points) == 1 or right_curve_points[-1][1] > vanishing_y + 40:
        right_top_x = int(center_x + width * 0.1)
        right_curve_points.append((right_top_x, vanishing_y))
    
    # Create a smooth polynomial fit for each curve if we have enough points
    if len(left_curve_points) >= 3:
        left_x = [p[0] for p in left_curve_points]
        left_y = [p[1] for p in left_curve_points]
        # Generate smooth curve with more points
        smooth_left_y = np.linspace(min(left_y), max(left_y), 10)
        z = np.polyfit(left_y, left_x, 2)  # Quadratic fit (degree 2)
        smooth_left_x = np.polyval(z, smooth_left_y)
        left_curve_points = list(zip(smooth_left_x.astype(int), smooth_left_y.astype(int)))
    
    if len(right_curve_points) >= 3:
        right_x = [p[0] for p in right_curve_points]
        right_y = [p[1] for p in right_curve_points]
        # Generate smooth curve with more points
        smooth_right_y = np.linspace(min(right_y), max(right_y), 10)
        z = np.polyfit(right_y, right_x, 2)  # Quadratic fit (degree 2)
        smooth_right_x = np.polyval(z, smooth_right_y)
        right_curve_points = list(zip(smooth_right_x.astype(int), smooth_right_y.astype(int)))
    
    # Create sidewalk visualization
    overlay = output.copy()
    
    # Create sidewalk polygon from curve points
    sidewalk_points = []
    for point in left_curve_points:
        sidewalk_points.append([point[0], point[1]])
    for point in reversed(right_curve_points):
        sidewalk_points.append([point[0], point[1]])
    
    # Convert to proper format for fillPoly
    sidewalk_poly = np.array([sidewalk_points], dtype=np.int32)
    
    # Draw the filled sidewalk
    cv2.fillPoly(overlay, sidewalk_poly, (0, 150, 0))
    
    # Add distance markers on the sidewalk with perspective
    for i in range(1, 6):
        y_pos = height - int((height - vanishing_y) * i / 5)
        # Find the x positions at this y-level by interpolating the curve points
        left_x = None
        right_x = None
        
        for j in range(len(left_curve_points) - 1):
            if left_curve_points[j][1] >= y_pos >= left_curve_points[j+1][1]:
                y1, y2 = left_curve_points[j][1], left_curve_points[j+1][1]
                x1, x2 = left_curve_points[j][0], left_curve_points[j+1][0]
                # Linear interpolation
                left_x = int(x1 + (y_pos - y1) * (x2 - x1) / (y2 - y1 + 1e-6))
                break
        
        for j in range(len(right_curve_points) - 1):
            if right_curve_points[j][1] >= y_pos >= right_curve_points[j+1][1]:
                y1, y2 = right_curve_points[j][1], right_curve_points[j+1][1]
                x1, x2 = right_curve_points[j][0], right_curve_points[j+1][0]
                # Linear interpolation
                right_x = int(x1 + (y_pos - y1) * (x2 - x1) / (y2 - y1 + 1e-6))
                break
        
        # If we found valid x positions, draw the distance marker
        if left_x is not None and right_x is not None:
            cv2.line(overlay, (left_x, y_pos), (right_x, y_pos), 
                     (255, 255, 255), max(1, 3 - i//2))
    
    # Add the overlay with transparency
    cv2.addWeighted(overlay, 0.4, output, 0.6, 0, output)
    
    # Draw the curved sidewalk boundaries
    for i in range(len(left_curve_points) - 1):
        cv2.line(output, 
                 (left_curve_points[i][0], left_curve_points[i][1]), 
                 (left_curve_points[i+1][0], left_curve_points[i+1][1]), 
                 (0, 0, 255), 3)
    
    for i in range(len(right_curve_points) - 1):
        cv2.line(output, 
                 (right_curve_points[i][0], right_curve_points[i][1]), 
                 (right_curve_points[i+1][0], right_curve_points[i+1][1]), 
                 (0, 0, 255), 3)
    
    # Add debug overlay if needed
    cv2.addWeighted(debug_image, 0.3, output, 0.7, 0, output)
    
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