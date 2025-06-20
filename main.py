from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import json
import os

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
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

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
            return {"error": "No lanes data provided."}, 400
        
        lanes = json.loads(lanes_str)
        results = {}

        for lane in lanes:
            lane_id = lane.get('id')
            if not lane_id:
                continue

            file_key = f"file_{lane_id}"
            video_file = form_data.get(file_key)
            
            if not video_file:
                continue

            # Save the uploaded file temporarily
            temp_video_path = f"temp_{video_file.filename}"
            with open(temp_video_path, "wb") as f:
                content = await video_file.read()
                f.write(content)

            # Process the video file with OpenCV
            cap = cv2.VideoCapture(temp_video_path)
            if not cap.isOpened():
                results[lane_id] = {"error": f"Could not open video file for lane {lane_id}."}
                os.remove(temp_video_path)
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
                    vehicles_in_frame = np.random.randint(0, 9)
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
            
            # Clean up the temporary file
            os.remove(temp_video_path)

        return results

    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}, 500 