from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import json
import os
import traceback
import logging
import requests
from zipfile import ZipFile
from io import BytesIO

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

# --- Vehicle Detection Setup (OpenCV DNN) ---

MODEL_URL = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel"
PROTOTXT_URL = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.prototxt.txt"
MODEL_PATH = "MobileNetSSD_deploy.caffemodel"
PROTOTXT_PATH = "MobileNetSSD_deploy.prototxt.txt"

# Define the classes the model can detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
VEHICLE_CLASSES = ["bus", "car", "motorbike", "train"]

def download_file(url, local_filename):
    """Downloads a file from a URL to a local file."""
    if os.path.exists(local_filename):
        logger.info(f"File '{local_filename}' already exists. Skipping download.")
        return True
    logger.info(f"Downloading '{local_filename}' from {url}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        logger.info(f"Successfully downloaded '{local_filename}'.")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download {url}. Error: {e}")
        return False

# Download model files before loading
model_downloaded = download_file(MODEL_URL, MODEL_PATH)
prototxt_downloaded = download_file(PROTOTXT_URL, PROTOTXT_PATH)

# Load the lightweight pre-trained model
try:
    if not model_downloaded or not prototxt_downloaded:
        raise FileNotFoundError("Required model files are missing and could not be downloaded.")
    net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
    logger.info("--- OpenCV DNN model loaded successfully. ---")
except Exception as e:
    raise RuntimeError(f"--- Fatal: Failed to load OpenCV DNN model. Details: {e} ---")

def detect_vehicles(frame):
    """Detects vehicles in a single frame using the OpenCV DNN model."""
    if frame is None:
        logger.warning("--- detect_vehicles received a None frame. ---")
        return 0
    
    (h, w) = frame.shape[:2]
    # Create a blob from the image and perform a forward pass of the model
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    vehicle_count = 0
    # Loop over the detections
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2: # Confidence threshold
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] in VEHICLE_CLASSES:
                vehicle_count += 1
                
    logger.info(f"--- Found {vehicle_count} vehicles in frame. ---")
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
        
        lanes = json.loads(lanes_str)
        results = {}

        for lane in lanes:
            lane_id = lane.get('id')
            if not lane_id:
                logger.warning("--- Skipping a lane because it has no ID. ---")
                continue

            file_key = f"file_{lane_id}"
            video_file = form_data.get(file_key)
            
            if not hasattr(video_file, 'filename'):
                logger.warning(f"--- File-like object not found for key '{file_key}'. ---")
                continue

            logger.info(f"--- Processing file '{video_file.filename}' for lane {lane_id}. ---")
            temp_video_path = f"temp_{video_file.filename}"
            try:
                with open(temp_video_path, "wb") as f:
                    content = await video_file.read()
                    f.write(content)

                cap = cv2.VideoCapture(temp_video_path)
                if not cap.isOpened():
                    logger.error(f"--- Could not open video file for lane {lane_id}. ---")
                    continue

                vehicle_count_total = 0
                frame_count = 0
                processed_frames_count = 0
                max_frames_to_process = 5 # Limit to prevent timeouts

                while cap.isOpened() and processed_frames_count < max_frames_to_process:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process every 15th frame to speed up analysis
                    if frame_count % 15 == 0:
                        vehicles_in_frame = detect_vehicles(frame)
                        vehicle_count_total += vehicles_in_frame
                        processed_frames_count += 1
                    frame_count += 1
                cap.release()
                
                results[lane_id] = { "vehicle_count": vehicle_count_total }
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