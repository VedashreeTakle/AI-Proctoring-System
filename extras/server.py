import threading
import cv2
import time
import numpy as np
import torch
import speech_recognition as sr
import mediapipe as mp
import os
import datetime
from ultralytics import YOLO
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device
import pyaudio
import wave
from flask import Flask, request, jsonify, Response
import requests
import json

app = Flask(__name__)

# -------------------- Global Variables -------------------- #
OTHER_FLASK_APP_URL = "http://localhost:5000/receive_proctoring_data"
evidence_dir = "cheating_evidence"

proctoring_running = False


# Camera access management
camera_lock = threading.Lock()
current_frame = None
camera_thread_running = False
camera_thread = None

# -------------------- Folder Setup -------------------- #
os.makedirs(evidence_dir, exist_ok=True)
os.makedirs(os.path.join(evidence_dir, "screenshots"), exist_ok=True)
os.makedirs(os.path.join(evidence_dir, "audio"), exist_ok=True)

# -------------------- Model Initialization -------------------- #
# YOLOv8 - face spoofing + mobile detection
face_model = YOLO("yolov8n.pt")     # Small YOLOv8 for spoof
mobile_model = YOLO("yolov8s.pt")   # Slightly stronger YOLOv8 for mobile detection
FACE_CONF_THRESHOLD = 0.5
MOBILE_CONF_THRESHOLD = 0.6

# YOLOv5 - person detection
device = select_device('cuda' if torch.cuda.is_available() else 'cpu')
people_model = DetectMultiBackend("yolov5s.pt", device=device)
people_model.warmup(imgsz=(1, 3, 640, 640))

# Mediapipe - face mesh for head pose and lips
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Speech recognizer
recognizer = sr.Recognizer()
cheating_keywords = ["answer", "solution", "help", "cheat", "google", "search", "tell me"]

# Head pose model points
model_points = np.array([
    (0.0, 0.0, 0.0),        # Nose
    (-30.0, -125.0, -30.0), # Left eye
    (30.0, -125.0, -30.0),  # Right eye
    (-60.0, -70.0, -60.0),  # Left mouth
    (60.0, -70.0, -60.0),   # Right mouth
    (0.0, -150.0, -25.0)    # Chin
], dtype="double")

# Lip landmarks
UPPER_LIP = [13, 14, 15, 16, 17]
LOWER_LIP = [82, 81, 80, 191, 178]
prev_lip_distance = None
lip_movement_threshold = 0.003

# Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5
is_recording_audio = False
audio_recording_thread = None

# -------------------- Camera Thread Function -------------------- #
def camera_capture_thread():
    """Thread function to continuously capture frames from camera"""
    global current_frame, camera_thread_running
    
    print("Starting camera thread...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Failed to open webcam")
        camera_thread_running = False
        return
        
    camera_thread_running = True
    
    try:
        while camera_thread_running:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to capture frame")
                break
                
            # Update the current frame for other functions to use
            with camera_lock:
                current_frame = frame.copy()
            
            time.sleep(0.01)  # Small sleep to reduce CPU usage
    finally:
        cap.release()
        print("Camera thread stopped")


# -------------------- Evidence Collection Functions -------------------- #
def capture_screenshot(reason):
    """Capture and save a screenshot when cheating is detected"""
    with camera_lock:
        if current_frame is not None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(evidence_dir, "screenshots", f"{reason}_{timestamp}.jpg")
            
            # Debug print to verify path
            print(f"Attempting to save screenshot to: {filename}")
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # Save the image with error handling
            try:
                cv2.imwrite(filename, current_frame)
                print(f"📸 Screenshot saved: {filename}")
                return filename
            except Exception as e:
                print(f"Error saving screenshot: {e}")
                return None
        else:
            print("No current frame to capture")
            return None

def collect_evidence(reason):
    """Collect both screenshot and audio evidence"""
    print(f"Collecting evidence for: {reason}")
    
    # Capture screenshot with error handling
    screenshot_path = capture_screenshot(reason)
    
    # Start audio recording
    start_audio_recording(reason)
    
    # Log evidence collection result
    evidence_data = {
        "reason": reason,
        "timestamp": datetime.datetime.now().isoformat(),
        "screenshot": screenshot_path
    }
    
    # Save evidence metadata to a log file
    try:
        log_path = os.path.join(evidence_dir, "evidence_log.txt")
        with open(log_path, "a") as f:
            f.write(f"{datetime.datetime.now().isoformat()} - {reason}: {screenshot_path}\n")
    except Exception as e:
        print(f"Error logging evidence: {e}")
    
    # Send data to other Flask app
    try:
        requests.post(
            OTHER_FLASK_APP_URL,
            json=evidence_data,
            timeout=1  # Short timeout to avoid blocking
        )
    except requests.exceptions.RequestException as e:
        print(f"Failed to send data to other Flask app: {e}")
        
    return evidence_data

def start_audio_recording(reason):
    """Start recording audio when cheating is detected"""
    global is_recording_audio, audio_recording_thread
    
    if is_recording_audio:
        return
        
    is_recording_audio = True
    audio_recording_thread = threading.Thread(
        target=record_audio, 
        args=(reason,)
    )
    audio_recording_thread.start()

def record_audio(reason):
    """Record audio for a set duration"""
    global is_recording_audio
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(evidence_dir, "audio", f"{reason}_{timestamp}.wav")
    
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    
    print(f"🎙️ Recording audio evidence for {RECORD_SECONDS} seconds...")
    frames = []
    
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    print("✅ Audio recording complete")
    
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    # Save the audio file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    print(f"🔊 Audio saved: {filename}")
    is_recording_audio = False
    return filename

# -------------------- Detection Functions -------------------- #
def detect_people(frame):
    """Detect number of people using YOLOv5"""
    img = cv2.resize(frame, (640, 640))[:, :, ::-1].transpose(2, 0, 1)
    img = torch.from_numpy(np.ascontiguousarray(img)).to(device).float() / 255.0
    img = img.unsqueeze(0)

    with torch.no_grad():
        pred = people_model(img)
    pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.45)

    count = 0
    for det in pred:
        if det is not None and len(det):
            for *xyxy, conf, cls in det:
                if int(cls) == 0:
                    count += 1
                    xyxy = list(map(int, xyxy))
                    cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
    return frame, count

def detect_cheating(text):
    return any(keyword in text.lower() for keyword in cheating_keywords)

def rotation_vector_to_euler_angles(rotation_vector):
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    pitch = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1]*2 + rotation_matrix[2, 2]*2))
    roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)

def get_head_pose(frame, landmarks):
    image_points = np.array([
        (landmarks[1][0], landmarks[1][1]),
        (landmarks[33][0], landmarks[33][1]),
        (landmarks[263][0], landmarks[263][1]),
        (landmarks[61][0], landmarks[61][1]),
        (landmarks[291][0], landmarks[291][1]),
        (landmarks[152][0], landmarks[152][1])
    ], dtype="double")

    size = frame.shape
    focal_length = size[1]
    center = (size[1] // 2, size[0] // 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, _ = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    return rotation_vector

def calculate_lip_distance(landmarks):
    return np.mean([
        np.linalg.norm(
            np.array([landmarks[UPPER_LIP[i]].x, landmarks[UPPER_LIP[i]].y]) -
            np.array([landmarks[LOWER_LIP[i]].x, landmarks[LOWER_LIP[i]].y])
        ) for i in range(len(UPPER_LIP))
    ])

# -------------------- Threaded Tasks -------------------- #
def process_audio():
    with sr.Microphone() as source:
        while True:
            try:
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source, timeout=5)
                text = recognizer.recognize_google(audio)
                print(f"🗣 Heard: {text}")
                if detect_cheating(text):
                    print("🚨 Cheating detected via speech!")
                    collect_evidence("speech_cheating")
            except Exception as e:
                print(f"🔇 Audio error: {e}")
            time.sleep(1)

def process_video():
    global prev_lip_distance
    
    # Make sure camera thread is running
    ensure_camera_running()
    
    while True:
        with camera_lock:
            if current_frame is None:
                time.sleep(0.1)
                continue
            
            # Create a copy to work with
            frame = current_frame.copy()
        
        # Person detection
        frame, people_count = detect_people(frame)
        if people_count > 1:
            print(f"🚨 Multiple people detected: {people_count}")
            collect_evidence("multiple_people")

        # Mobile phone detection
        mobile_results = mobile_model(frame)
        for result in mobile_results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                conf = box.conf[0].item()
                if class_id == 67 and conf > MOBILE_CONF_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"📱 Mobile ({conf:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    print("🚨 Mobile phone detected!")
                    if conf > 0.5:
                        print(f"🚨 Mobile Detected — Cheating!")
                        collect_evidence("mobile_detected")

        # Face spoof detection
        results = face_model.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), verbose=False)
        for r in results:
            for box in r.boxes:
                conf = box.conf[0].item()
                if conf < FACE_CONF_THRESHOLD:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = "Spoof" if int(box.cls[0]) == 1 else "Real"
                color = (0, 0, 255) if label == "Spoof" else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if label == "Spoof":
                    print("🚨 Spoof detected!")
                    collect_evidence("face_spoof")
                elif conf < 0.85:
                    print(f"⚠ Possible Spoof Detected")

        # Head pose & lip movement
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks_px = [(int(pt.x * frame.shape[1]), int(pt.y * frame.shape[0])) for pt in face_landmarks.landmark]
                rotation_vector = get_head_pose(frame, landmarks_px)
                yaw, pitch, roll = rotation_vector_to_euler_angles(rotation_vector)
                if abs(yaw) > 20 or pitch < -10 or pitch > 15 or roll < 80 or roll > 110:
                    print("🚨 Cheating: Looking Away")
                    collect_evidence("looking_away")

                lip_distance = calculate_lip_distance(face_landmarks.landmark)
                if prev_lip_distance and abs(lip_distance - prev_lip_distance) > lip_movement_threshold:
                    print("🚨 Lip Movement Detected (Possible Talking)")
                    collect_evidence("lip_movement")
                prev_lip_distance = lip_distance

        cv2.imshow("AI Proctoring", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        time.sleep(0.03)  # Limit processing speed to reduce CPU usage

def ensure_camera_running():
    """Make sure the camera thread is running"""
    global camera_thread, camera_thread_running
    
    if camera_thread is None or not camera_thread.is_alive():
        camera_thread_running = False
        camera_thread = threading.Thread(target=camera_capture_thread)
        camera_thread.daemon = True
        camera_thread.start()
        # Give the camera thread time to start
        time.sleep(1)

# -------------------- Flask Routes -------------------- #
@app.route('/check_cheating', methods=['POST'])
def check_cheating():
    """API endpoint to start proctoring process"""
    try:
        # Make sure camera is running
        ensure_camera_running()
        
        # Start the proctoring in background threads
        video_thread = threading.Thread(target=process_video)
        audio_thread = threading.Thread(target=process_audio)
        
        video_thread.daemon = True
        audio_thread.daemon = True
        
        video_thread.start()
        audio_thread.start()
        
        return jsonify({
            "status": "success",
            "message": "Proctoring started successfully",
            "evidence_dir": evidence_dir
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to start proctoring: {str(e)}"
        }), 500

@app.route('/receive_proctoring_data', methods=['POST'])
def receive_proctoring_data():
    """Endpoint to receive proctoring data from other Flask app"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "status": "error",
                "message": "No data received"
            }), 400
            
        # Process received data (e.g., log it)
        print(f"Received proctoring data: {data}")
        
        return jsonify({
            "status": "success",
            "message": "Data received successfully"
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error processing data: {str(e)}"
        }), 500

@app.route('/video_feed')
def video_feed():
    """Route to serve video feed for web viewing"""
    def generate_frames():
        while True:
            with camera_lock:
                if current_frame is None:
                    time.sleep(0.1)
                    continue
                    
                frame = current_frame.copy()
            
            # Encode frame as JPEG for streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
                
            # Convert to bytes and yield for Flask response
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.03)  # Limit frame rate
    
    return Response(
        generate_frames(), 
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# -------------------- Main -------------------- #
def main():
    # Start the camera thread
    ensure_camera_running()
    
    # Start processing threads
    video_thread = threading.Thread(target=process_video)
    audio_thread = threading.Thread(target=process_audio)
    
    video_thread.daemon = True
    audio_thread.daemon = True
    
    video_thread.start()
    audio_thread.start()
    
    global proctoring_running
    
    print("Starting proctoring server...")
    print(f"Waiting for trigger from {OTHER_FLASK_APP_URL}")
    
    # Don't automatically start proctoring - wait for the trigger
    proctoring_running = False
    
    # Run the Flask app to listen for the trigger
    app.run(debug=False, host='127.0.0.1', port=5007)

if __name__ == "__main__":
    main()