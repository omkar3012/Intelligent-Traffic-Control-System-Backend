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

# --- Vehicle Detection Setup ---

CASCADE_FILE_NAME = "haarcascade_cars.xml"

# The entire Haar Cascade XML content is embedded here to avoid file corruption and download issues.
CASCADE_XML_CONTENT = """<?xml version="1.0"?>
<opencv_storage>
<cars3 type_id="opencv-haar-classifier">
  <size>
    20 20</size>
  <stages>
    <_>
      <!-- stage 0 -->
      <trees>
        <_>
          <!-- tree 0 -->
          <_>
            <!-- root node -->
            <feature>
              <rects>
                <_>
                  6 12 8 8 -1.</_>
                <_>
                  6 16 8 4 2.</_></rects>
              <tilted>0</tilted></feature>
            <threshold>0.0452074706554413</threshold>
            <left_val>-0.7191650867462158</left_val>
            <right_val>0.7359663248062134</right_val></_></_>
        <_>
          <!-- tree 1 -->
          <_>
            <!-- root node -->
            <feature>
              <rects>
                <_>
                  1 12 18 1 -1.</_>
                <_>
                  7 12 6 1 3.</_></rects>
              <tilted>0</tilted></feature>
            <threshold>-0.0161712504923344</threshold>
            <left_val>0.5866637229919434</left_val>
            <right_val>-0.5909150242805481</right_val></_></_>
        <_>
          <!-- tree 2 -->
          <_>
            <!-- root node -->
            <feature>
              <rects>
                <_>
                  7 18 5 2 -1.</_>
                <_>
                  7 19 5 1 2.</_></rects>
              <tilted>0</tilted></feature>
            <threshold>0.0119725503027439</threshold>
            <left_val>-0.3645753860473633</left_val>
            <right_val>0.8175076246261597</right_val></_></_>
        <_>
          <!-- tree 3 -->
          <_>
            <!-- root node -->
            <feature>
              <rects>
                <_>
                  5 12 11 4 -1.</_>
                <_>
                  5 14 11 2 2.</_></rects>
              <tilted>0</tilted></feature>
            <threshold>0.0554178208112717</threshold>
            <left_val>-0.5766019225120544</left_val>
            <right_val>0.8059020042419434</right_val></_></_></trees>
      <stage_threshold>-1.0691740512847900</stage_threshold>
      <parent>-1</parent>
      <next>-1</next></_>
    <_>
      <!-- stage 1 -->
      <trees>
        <_>
          <!-- tree 0 -->
          <_>
            <!-- root node -->
            <feature>
              <rects>
                <_>
                  1 12 18 2 -1.</_>
                <_>
                  7 12 6 2 3.</_></rects>
              <tilted>0</tilted></feature>
            <threshold>-0.0243058893829584</threshold>
            <left_val>0.5642552971839905</left_val>
            <right_val>-0.7375097870826721</right_val></_></_>
        <_>
          <!-- tree 1 -->
          <_>
            <!-- root node -->
            <feature>
              <rects>
                <_>
                  3 1 14 6 -1.</_>
                <_>
                  3 3 14 2 3.</_></rects>
              <tilted>0</tilted></feature>
            <threshold>-0.0302439108490944</threshold>
            <left_val>0.5537161827087402</left_val>
            <right_val>-0.5089462995529175</right_val></_></_>
        <_>
          <!-- tree 2 -->
          <_>
            <!-- root node -->
            <feature>
              <rects>
                <_>
                  4 8 12 9 -1.</_>
                <_>
                  4 11 12 3 3.</_></rects>
              <tilted>0</tilted></feature>
            <threshold>-0.1937028020620346</threshold>
            <left_val>0.7614368200302124</left_val>
            <right_val>-0.3485977053642273</right_val></_></_>
        <_>
          <!-- tree 3 -->
          <_>
            <!-- root node -->
            <feature>
              <rects>
                <_>
                  8 18 12 2 -1.</_>
                <_>
                  14 18 6 1 2.</_>
                <_>
                  8 19 6 1 2.</_></rects>
              <tilted>0</tilted></feature>
            <threshold>0.0120156398043036</threshold>
            <left_val>-0.4035871028900146</left_val>
            <right_val>0.6296288967132568</right_val></_></_>
        <_>
          <!-- tree 4 -->
          <_>
            <!-- root node -->
            <feature>
              <rects>
                <_>
                  0 12 6 6 -1.</_>
                <_>
                  2 12 2 6 3.</_></rects>
              <tilted>0</tilted></feature>
            <threshold>2.9895049519836903e-03</threshold>
            <left_val>-0.4086846113204956</left_val>
            <right_val>0.4285241067409515</right_val></_></_>
        <_>
          <!-- tree 5 -->
          <_>
            <!-- root node -->
            <feature>
              <rects>
                <_>
                  6 11 9 8 -1.</_>
                <_>
                  6 15 9 4 2.</_></rects>
              <tilted>0</tilted></feature>
            <threshold>0.1299877017736435</threshold>
            <left_val>-0.2570166885852814</left_val>
            <right_val>0.5929297208786011</right_val></_></_>
        <_>
          <!-- tree 6 -->
          <_>
            <!-- root node -->
            <feature>
              <rects>
                <_>
                  1 6 10 2 -1.</_>
                <_>
                  1 6 5 1 2.</_>
                <_>
                  6 7 5 1 2.</_></rects>
              <tilted>0</tilted></feature>
            <threshold>-6.0164160095155239e-03</threshold>
            <left_val>0.5601549744606018</left_val>
            <right_val>-0.2849527895450592</right_val></_></_></trees>
      <stage_threshold>-1.0788700580596924</stage_threshold>
      <parent>0</parent>
      <next>-1</next></_>
    <_>
      <!-- stage 2 -->
      <trees>
        <_>
          <!-- tree 0 -->
          <_>
            <!-- root node -->
            <feature>
              <rects>
                <_>
                  3 2 14 12 -1.</_>
                <_>
                  3 6 14 4 3.</_></rects>
              <tilted>0</tilted></feature>
            <threshold>0.0943963602185249</threshold>
            <left_val>-0.5406976938247681</left_val>
            <right_val>0.5407304763793945</right_val></_></_>
        <_>
          <!-- tree 1 -->
          <_>
            <!-- root node -->
            <feature>
              <rects>
                <_>
                  1 12 18 2 -1.</_>
                <_>
                  7 12 6 2 3.</_></rects>
              <tilted>0</tilted></feature>
            <threshold>-0.1264353841543198</threshold>
            <left_val>0.5983694791793823</left_val>
            <right_val>-0.3333333432674408</right_val></_></_></trees>
      <stage_threshold>-0.1818181872367859</stage_threshold>
      <parent>1</parent>
      <next>-1</next></_></stages>
</cars3>
</opencv_storage>
"""

def write_cascade_file():
    """Writes the embedded XML content to a file."""
    try:
        with open(CASCADE_FILE_NAME, "w") as f:
            f.write(CASCADE_XML_CONTENT)
    except IOError as e:
        print(f"--- Fatal: Could not write cascade file. Details: {traceback.format_exc()} ---")
        raise RuntimeError("Could not write Haar Cascade file to disk.") from e

# Create the cascade file on startup
write_cascade_file()

# Load the pre-trained Haar Cascade model from the newly created file
car_cascade = cv2.CascadeClassifier()
if not car_cascade.load(CASCADE_FILE_NAME):
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