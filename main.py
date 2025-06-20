from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import json
import os
import traceback

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

# --- Vehicle Detection ---
# Load the pre-trained Haar Cascade model for car detection
car_cascade = cv2.CascadeClassifier()
cascade_path = os.path.join(os.path.dirname(__file__), 'haarcascade_cars.xml')
if not car_cascade.load(cascade_path):
    print("--- Error: Failed to load car cascade classifier ---")
    # You might want to handle this more gracefully, e.g., by raising an exception
    # that gets caught by a startup event handler in a real application.

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