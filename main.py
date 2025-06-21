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
import logging

# --- Logging Setup ---
# This will ensure logs are printed in the Render environment.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Construct an absolute path to the model file to ensure it's found in any environment.
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'yolov8n.pt')

# Load the YOLOv8 model from the local path.
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"--- Fatal: YOLOv8 model not found at {MODEL_PATH} ---")
    model = YOLO(MODEL_PATH)
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
    logger.info("--- Received request for /process-video ---")
    try:
        form_data = await request.form()
        logger.info(f"--- Form keys received: {list(form_data.keys())} ---")
        
        lanes_str = form_data.get("lanes")
        if not lanes_str:
            logger.error("--- 'lanes' data not found in form. ---")
            return JSONResponse(status_code=400, content={"error": "No lanes data provided."})
        
        logger.info(f"--- 'lanes' string received: {lanes_str} ---")
        lanes = json.loads(lanes_str)
        results = {}

        logger.info(f"--- Parsed {len(lanes)} lanes. Starting iteration... ---")
        for lane in lanes:
            lane_id = lane.get('id')
            logger.info(f"--- Processing lane with ID: {lane_id} ---")
            if not lane_id:
                logger.warning("--- Skipping a lane because it has no ID. ---")
                continue

            file_key = f"file_{lane_id}"
            logger.info(f"--- Attempting to find file with key: '{file_key}' ---")
            video_file = form_data.get(file_key)
            
            if not isinstance(video_file, UploadFile):
                logger.warning(f"--- File not found for key '{file_key}'. Type received: {type(video_file)} ---")
                continue

            logger.info(f"--- Successfully found file '{video_file.filename}' for lane {lane_id}. ---")
            temp_video_path = f"temp_{video_file.filename}"
            try:
                with open(temp_video_path, "wb") as f:
                    content = await video_file.read()
                    f.write(content)
                
                logger.info(f"--- Video for lane {lane_id} saved to temp file. Starting OpenCV processing. ---")
                cap = cv2.VideoCapture(temp_video_path)
                if not cap.isOpened():
                    logger.error(f"--- Could not open video file for lane {lane_id}. ---")
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
                average_vehicles = vehicle_count_total / max(processed_frames, 1)

                results[lane_id] = {
                    "vehicle_count": vehicle_count_total,
                    "average_vehicles": round(average_vehicles, 2),
                    "processed_frames": processed_frames,
                    "total_frames": total_frames
                }
                logger.info(f"--- Successfully processed lane {lane_id}. Vehicle count: {vehicle_count_total} ---")
            finally:
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
        
        logger.info(f"--- Finished processing all lanes. Returning results: {results} ---")
        return JSONResponse(content=results)

    except Exception as e:
        detailed_error = traceback.format_exc()
        logger.error(f"--- An unhandled error occurred in /process-video: {detailed_error} ---")
        return JSONResponse(
            status_code=500,
            content={"error": "An internal server error occurred.", "details": detailed_error}
        ) 