from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import json
import os
import traceback
import requests

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

CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_car.xml"
CASCADE_FILE_NAME = "haarcascade_cars.xml"

def download_cascade_file():
    """Downloads the Haar Cascade file if it doesn't already exist."""
    if not os.path.exists(CASCADE_FILE_NAME):
        print(f"Downloading {CASCADE_FILE_NAME} from {CASCADE_URL}...")
        try:
            response = requests.get(CASCADE_URL, timeout=10)
            response.raise_for_status()  # Raise an exception for bad status codes
            with open(CASCADE_FILE_NAME, "wb") as f:
                f.write(response.content)
            print("Download complete.")
        except requests.RequestException as e:
            # Print the full exception to get more details in the logs
            print(f"--- Error downloading cascade file. Details: {traceback.format_exc()} ---")
            raise RuntimeError("Could not download Haar Cascade file for vehicle detection.") from e

# Download the file on startup
download_cascade_file()

# Load the pre-trained Haar Cascade model
car_cascade = cv2.CascadeClassifier()
if not car_cascade.load(CASCADE_FILE_NAME):
    # This error should now be highly unlikely
    raise RuntimeError(f"--- Fatal: Failed to load car cascade classifier from {CASCADE_FILE_NAME} ---")

def detect_vehicles(frame):
    """Detects vehicles in a single frame using the Haar Cascade model."""
    if frame is None:
        return 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Adjust scaleFactor and minNeighbors for detection sensitivity.
    # scaleFactor: How much the image size is reduced at each image scale.
    # minNeighbors: How many neighbors each candidate rectangle should have to retain it.
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(cars)

@app.post('/process-video')
async def process_video(request: Request):
    """
    This endpoint now accepts multipart/form-data, processes each video file,
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
                sample_interval = 30
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
                    "average_vehicles": average_vehicles,
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