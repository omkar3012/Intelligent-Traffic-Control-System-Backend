from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import json
import os
import traceback
from ultralytics import YOLO
import torch

app = FastAPI()

# --- CORS Configuration ---
# This allows your frontend on Vercel to communicate with this backend.
origins = [
    "https://intelligent-traffic-control-system.vercel.app",
    "http://localhost:3000",  # For local development
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    # This regex allows all Vercel preview deployments for your project.
    allow_origin_regex=r"https://intelligent-traffic-control-system-.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Vehicle Detection Setup ---

# Load the YOLOv8 model
# 'yolov8n.pt' is a small and fast model, suitable for CPU execution.
try:
    model = YOLO('yolov8n.pt')
    print("--- YOLOv8 model loaded successfully. ---")
except Exception as e:
    raise RuntimeError(f"--- Fatal: Failed to load YOLOv8 model. Details: {e} ---")

# Define the classes that should be considered as vehicles
VEHICLE_CLASSES = [2, 3, 5, 7]  # In COCO dataset: 2=car, 3=motorcycle, 5=bus, 7=truck

def detect_vehicles(frame):
    """Detects vehicles in a single frame using the YOLOv8 model."""
    if frame is None:
        print("--- detect_vehicles received a None frame. ---")
        return 0
    
    # --- Detailed Logging ---
    print(f"--- Processing frame of shape: {frame.shape}, dtype: {frame.dtype} ---")

    # Perform inference with a low confidence threshold to catch all potential detections
    results = model(frame, verbose=False, conf=0.1)
    
    all_detected_classes = [int(box.cls) for box in results[0].boxes]
    if all_detected_classes:
        print(f"--- YOLO detected objects with classes: {all_detected_classes} ---")
    else:
        print("--- YOLO detected no objects in this frame. ---")

    vehicle_count = 0
    # The result object contains bounding boxes, classes, and confidences
    for box in results[0].boxes:
        try:
            # Check if the detected object class is in our list of vehicle classes
            if int(box.cls) in VEHICLE_CLASSES:
                vehicle_count += 1
        except (ValueError, IndexError):
            # Ignore if class ID is not a valid integer or out of bounds
            continue
            
    print(f"--- Found {vehicle_count} vehicles in frame. ---")
    return vehicle_count

@app.post('/process-video')
async def process_video(request: Request):
    """
    This endpoint accepts multipart/form-data, processes each video file,
    and returns a JSON object with vehicle detection results for each lane.
    """
    try:
        form_data = await request.form()
        lanes_str = form_data.get("lanes")
        if not lanes_str:
            return JSONResponse(status_code=400, content={"error": "No lanes data provided."})
        
        lanes = json.loads(lanes_str)
        results = {}

        for lane in lanes:
            lane_id = lane.get('id')
            if not lane_id:
                continue

            file_key = f"file_{lane_id}"
            video_file = form_data.get(file_key)
            
            if not isinstance(video_file, UploadFile):
                continue

            temp_video_path = f"temp_{video_file.filename}"
            try:
                with open(temp_video_path, "wb") as f:
                    content = await video_file.read()
                    f.write(content)

                cap = cv2.VideoCapture(temp_video_path)
                if not cap.isOpened():
                    results[lane_id] = {"error": f"Could not open video file for lane {lane_id}."}
                    continue

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                sample_interval = 30  # Process 1 frame per second for a 30fps video
                vehicle_count_total = 0
                frame_count = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    if frame_count % sample_interval == 0:
                        vehicles_in_frame = detect_vehicles(frame)
                        vehicle_count_total += vehicles_in_frame
                cap.release()
                
                processed_frames = frame_count // sample_interval if sample_interval > 0 else 0
                # Use total detections instead of averaging for a cumulative count
                average_vehicles = vehicle_count_total / max(processed_frames, 1)

                results[lane_id] = {
                    "vehicle_count": vehicle_count_total,
                    "average_vehicles": round(average_vehicles, 2),
                    "processed_frames": processed_frames,
                    "total_frames": total_frames
                }
            finally:
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)

        return JSONResponse(content=results)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "An internal server error occurred.", "details": traceback.format_exc()}
        ) 