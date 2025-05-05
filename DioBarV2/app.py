# app.py - Web Interface for Raspberry Pi Rebar Analysis

from flask import Flask, render_template, request, jsonify, send_from_directory, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import requests
import json
import base64
import time
import threading
import cv2
import RPi.GPIO as GPIO
import numpy as np
import datetime
import glob
import io
import zipfile

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration
RASPI_IP = "localhost"  # Default IP for Raspberry Pi in WiFi direct mode
RASPI_PORT = 5000
RASPI_API_URL = f"http://{RASPI_IP}:{RASPI_PORT}/api"
POLLING_INTERVAL = 3  # Seconds between polling for new data

# Ultrasonic sensor pins
TRIG = 23
ECHO = 24

# GPIO setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

# Directory to save captured images
CAPTURE_DIR = "captured_images"
os.makedirs(CAPTURE_DIR, exist_ok=True)

# Local storage for the latest data
latest_data = {
    "connected": False,
    "last_image": None,
    "last_results": [],
    "last_update": None,
    "total_volume": 0,
    "distance": 0
}

# Camera setup
camera = None
frame_lock = threading.Lock()
latest_frame = None
distance_lock = threading.Lock()
latest_distance = 0
camera_running = False

def init_camera():
    """Initialize the camera"""
    global camera
    try:
        if camera is not None:
            camera.release()
        
        camera = cv2.VideoCapture(0)  # Default to camera 0
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not camera.isOpened():
            print("Failed to open camera")
            return False
            
        return True
    except Exception as e:
        print(f"Error initializing camera: {e}")
        return False

def measure_distance():
    """Measure distance using ultrasonic sensor"""
    # Send a 10us pulse to trigger
    GPIO.output(TRIG, False)
    time.sleep(0.01)
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)
    
    pulse_start = time.time()
    pulse_end = time.time()
    
    # Wait for echo to start (timeout after 1 second)
    timeout_start = time.time()
    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()
        if time.time() - timeout_start > 1:
            return -1  # Timeout error
    
    # Wait for echo to end (timeout after 1 second)
    timeout_start = time.time()
    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()
        if time.time() - timeout_start > 1:
            return -1  # Timeout error
    
    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150  # speed of sound / 2
    return round(distance, 2)

def distance_monitor():
    """Thread to continuously monitor distance"""
    global latest_distance
    
    while camera_running:
        dist = measure_distance()
        if dist > 0:
            with distance_lock:
                latest_distance = dist
                latest_data["distance"] = dist
        time.sleep(0.5)  # Check distance every 0.5 seconds

def capture_frames():
    """Thread to continuously capture frames"""
    global latest_frame, camera_running
    
    if not init_camera():
        print("Failed to initialize camera for frame capture")
        return
    
    print("Starting frame capture thread")
    
    while camera_running:
        try:
            ret, frame = camera.read()
            if not ret:
                print("Failed to capture frame, reinitializing camera...")
                if not init_camera():
                    time.sleep(1)
                    continue
                ret, frame = camera.read()
                if not ret:
                    continue
                    
            # Store the latest frame for streaming
            with frame_lock:
                latest_frame = frame.copy()
                
            # Brief sleep to reduce CPU usage
            time.sleep(0.03)  # ~30 fps
            
        except Exception as e:
            print(f"Error in frame capture: {e}")
            time.sleep(0.5)
    
    # Clean up when thread exits
    if camera is not None:
        camera.release()
    print("Frame capture thread stopped")

def generate_frames():
    """Generate frames for video streaming"""
    global latest_frame
    
    while camera_running:
        # Get the latest frame with thread safety
        with frame_lock:
            if latest_frame is not None:
                frame = latest_frame.copy()
            else:
                time.sleep(0.1)
                continue
        
        try:
            # Add distance overlay to streaming video
            with distance_lock:
                dist = latest_distance
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f"Distance: {dist} cm", (10, 30), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            
            # If analysis results are available, show volume too
            if latest_data["total_volume"] > 0:
                cv2.putText(frame, f"Volume: {latest_data['total_volume']:.2f} cc", (10, 60), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Encode the frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # Yield the frame in the format expected by Flask
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Error generating frame: {e}")
            time.sleep(0.1)

def check_connection():
    """Check if we can connect to the Raspberry Pi"""
    try:
        # First check if API is reachable
        response = requests.get(f"{RASPI_API_URL}/status", timeout=2)
        if response.status_code == 200:
            # Verify we can actually get data from it
            try:
                data_response = requests.get(f"{RASPI_API_URL}/latest", timeout=2)
                if data_response.status_code == 200:
                    print("Successfully connected to Raspberry Pi API")
                    return True
            except Exception as e:
                print(f"Could connect to API but failed to get data: {e}")
        return False
    except Exception as e:
        print(f"Failed to connect to Raspberry Pi API: {e}")
        return False

def get_raspi_data():
    """Poll the Raspberry Pi for new data"""
    global latest_data
    
    while True:
        connected = check_connection()
        old_connected = latest_data.get("connected", False)
        latest_data["connected"] = connected
        
        # Always emit connection status if it changed
        if connected != old_connected:
            print(f"Connection status changed from {old_connected} to {connected}")
            socketio.emit("connection_status", {"connected": connected})
        
        if connected:
            try:
                # Get the latest analysis results
                response = requests.get(f"{RASPI_API_URL}/latest", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("timestamp") != latest_data.get("last_update"):
                        print(f"New data received with timestamp: {data.get('timestamp')}")
                        latest_data["last_update"] = data.get("timestamp")
                        latest_data["last_results"] = data.get("segments", [])
                        latest_data["total_volume"] = data.get("total_volume", 0)
                        
                        # Get the latest image if available
                        if data.get("image_available", False):
                            img_response = requests.get(f"{RASPI_API_URL}/latest_image", timeout=5)
                            if img_response.status_code == 200:
                                latest_data["last_image"] = img_response.json().get("image")
                        
                        # Notify clients about new data
                        socketio.emit("new_data", {
                            "connected": True,
                            "timestamp": latest_data["last_update"],
                            "has_image": latest_data["last_image"] is not None,
                            "segments_count": len(latest_data["last_results"]),
                            "total_volume": latest_data["total_volume"],
                            "distance": latest_data["distance"]
                        })
                
                # Always emit connection status periodically
                socketio.emit("connection_status", {"connected": True})
            except Exception as e:
                print(f"Error polling Raspberry Pi: {e}")
                socketio.emit("connection_error", {"error": str(e)})
                socketio.emit("connection_status", {"connected": False})
        else:
            socketio.emit("connection_status", {"connected": False})
        
        # Wait before polling again
        time.sleep(POLLING_INTERVAL)

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/connection_status')
def connection_status():
    return jsonify({
        "connected": latest_data["connected"],
        "last_update": latest_data["last_update"]
    })

@app.route('/api/latest_data')
def get_latest_data():
    return jsonify({
        "connected": latest_data["connected"],
        "timestamp": latest_data["last_update"],
        "segments": latest_data["last_results"],
        "total_volume": latest_data["total_volume"],
        "has_image": latest_data["last_image"] is not None,
        "distance": latest_data["distance"]
    })

@app.route('/api/latest_image')
def get_latest_image():
    if latest_data["last_image"]:
        return jsonify({"image": latest_data["last_image"]})
    return jsonify({"error": "No image available"}), 404

@app.route('/api/distance')
def get_distance():
    """Get the current distance reading"""
    with distance_lock:
        return jsonify({"distance": latest_distance})

@app.route('/api/trigger_capture', methods=["POST"])
def trigger_capture():
    if not latest_data["connected"]:
        return jsonify({"error": "Not connected to Raspberry Pi"}), 503
    
    try:
        # Add timeout to prevent hanging
        response = requests.post(f"{RASPI_API_URL}/capture", timeout=10)
        if response.status_code == 200:
            return jsonify({"message": "Capture triggered successfully"})
        return jsonify({"error": f"Error: {response.text}"}), response.status_code
    except Exception as e:
        return jsonify({"error": f"Connection error: {str(e)}"}), 500

@app.route('/api/set_config', methods=["POST"])
def set_config():
    if not latest_data["connected"]:
        return jsonify({"error": "Not connected to Raspberry Pi"}), 503
    
    try:
        # Forward the configuration to the Raspberry Pi
        config_data = request.json
        response = requests.post(
            f"{RASPI_API_URL}/config", 
            json=config_data,
            timeout=5
        )
        if response.status_code == 200:
            return jsonify({"message": "Configuration updated successfully"})
        return jsonify({"error": f"Error: {response.text}"}), response.status_code
    except Exception as e:
        return jsonify({"error": f"Connection error: {str(e)}"}), 500

@app.route('/api/get_config')
def get_config():
    if not latest_data["connected"]:
        return jsonify({"error": "Not connected to Raspberry Pi"}), 503
    
    try:
        response = requests.get(f"{RASPI_API_URL}/config", timeout=5)
        if response.status_code == 200:
            return jsonify(response.json())
        return jsonify({"error": f"Error: {response.text}"}), response.status_code
    except Exception as e:
        return jsonify({"error": f"Connection error: {str(e)}"}), 500

@app.route('/api/capture_local', methods=["POST"])
def capture_local_image():
    """Capture an image and save it locally with distance and volume overlay"""
    global latest_frame
    
    try:
        # Get the current timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get the current distance reading
        with distance_lock:
            dist = latest_distance
        
        # Get the latest frame with thread safety
        with frame_lock:
            if latest_frame is not None:
                frame = latest_frame.copy()
            else:
                return jsonify({"success": False, "message": "No frame available"})
        
        # Add distance and volume overlays to the captured image
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"Distance: {dist} cm", (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        if latest_data["total_volume"] > 0:
            cv2.putText(frame, f"Volume: {latest_data['total_volume']:.2f} cc", (10, 70), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Save the image
        filename = f"capture_{timestamp}_{dist}cm.jpg"
        filepath = os.path.join(CAPTURE_DIR, filename)
        cv2.imwrite(filepath, frame)
        
        # Convert image to base64 for immediate display
        _, buffer = cv2.imencode('.jpg', frame)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "success": True, 
            "message": f"Image saved as {filename}",
            "image": image_base64,
            "filepath": filepath,
            "timestamp": timestamp,
            "distance": dist,
            "volume": latest_data["total_volume"]
        })
        
    except Exception as e:
        return jsonify({"success": False, "message": f"Error: {str(e)}"})

@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve captured images"""
    return send_from_directory(CAPTURE_DIR, filename)

@app.route('/api/image_list')
def get_image_list():
    """Get a list of all captured images"""
    images = []
    for file in sorted(glob.glob(os.path.join(CAPTURE_DIR, "*.jpg")), reverse=True):
        filename = os.path.basename(file)
        images.append({
            "filename": filename,
            "url": f"/images/{filename}",
            "timestamp": os.path.getmtime(file)
        })
    return jsonify({"images": images})

@socketio.on('connect')
def socket_connect():
    print("Client connected via Socket.IO")
    emit('connection_status', {
        "connected": latest_data["connected"],
        "last_update": latest_data["last_update"]
    })

@socketio.on('disconnect')
def socket_disconnect():
    print("Client disconnected from Socket.IO")

def cleanup():
    """Clean up resources when app exits"""
    global camera_running
    camera_running = False
    if camera is not None:
        camera.release()
    GPIO.cleanup()

if __name__ == "__main__":
    print("Starting RebarVista Web Interface...")
    print(f"Connecting to Raspberry Pi at {RASPI_API_URL}")
    
    # Initialize camera system
    camera_running = True
    
    # Start the frame capture thread
    frame_thread = threading.Thread(target=capture_frames, daemon=True)
    frame_thread.start()
    
    # Start distance monitoring thread
    distance_thread = threading.Thread(target=distance_monitor, daemon=True)
    distance_thread.start()
    
    # Start the data polling thread
    polling_thread = threading.Thread(target=get_raspi_data, daemon=True)
    polling_thread.start()
    
    try:
        # Start the Flask-SocketIO server
        socketio.run(app, host='0.0.0.0', port=8000, debug=True, use_reloader=False)
    finally:
        cleanup()