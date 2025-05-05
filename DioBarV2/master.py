#!/usr/bin/env python3
# Optimized Rebar Analysis Application for Raspberry Pi 5 with API server integration
# Now with external camera support and web interface integration

import os
import time
import torch
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import threading
import json
import math
import csv
import traceback
import gc  # Garbage collector
import base64
import io
from flask import Flask, jsonify, request, Response
from flask_cors import CORS

# Import Detectron2 libraries
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

# GPIO setup for distance sensor if available
try:
    import RPi.GPIO as GPIO
    # Ultrasonic sensor pins
    TRIG = 23
    ECHO = 24
    
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(TRIG, GPIO.OUT)
    GPIO.setup(ECHO, GPIO.IN)
    USE_DISTANCE_SENSOR = True
except ImportError:
    print("GPIO library not available - distance sensor functionality disabled")
    USE_DISTANCE_SENSOR = False

# Global state for API
latest_analysis = {
    "timestamp": None,
    "image": None,
    "image_path": None,
    "segments": [],
    "total_volume": 0,
    "distance": 0
}

# Initialize Flask API
api_app = Flask(__name__)
CORS(api_app)

# Global reference to app instance
app_instance = None

# Camera and distance measurement variables
latest_frame = None
frame_lock = threading.Lock()
latest_distance = 0
distance_lock = threading.Lock()
camera_running = True  # Flag to control camera thread

class RebarAnalysisApp:
    def __init__(self):
        # Initialize variables
        self.captured_frame = None
        self.is_processing = False  # Flag to prevent multiple captures
        self.result_image = None  # Store processed image for API
        self.camera = None  # Camera reference
        
        # Define colors
        self.colors = {
            'primary': '#3498db',      # Blue
            'accent': '#27ae60',       # Green
            'warning': '#e74c3c',      # Red
            'bg': '#f5f5f5',           # Light background
            'dark': '#2c3e50',         # Dark blue/gray
            'text': '#34495e',         # Dark text
            'light_text': '#ffffff'    # Light text
        }
        
        # Create results directory if it doesn't exist
        self.results_dir = "analysis_results"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        
        # Camera settings
        self.camera_index = 0  # Default to first camera
        self.load_camera_settings()
        
        # Load models and cement ratios
        try:
            self.load_models()
            self.load_cement_ratios()
            print("Models loaded successfully")
        except Exception as e:
            print(f"Initialization error: {e}")
            print(traceback.format_exc())
        
        # Initialize camera
        try:
            self.initialize_camera()
            print("Camera initialized")
        except Exception as e:
            print(f"Camera error: {e}")
            print(traceback.format_exc())
    
    def load_camera_settings(self):
        """Load camera settings from file"""
        try:
            if os.path.exists('camera_settings.json'):
                with open('camera_settings.json', 'r') as f:
                    settings = json.load(f)
                    self.camera_index = settings.get('camera_index', 0)
                    print(f"Loaded camera index: {self.camera_index}")
        except Exception as e:
            print(f"Error loading camera settings: {e}")
            # Keep using default settings
    
    def save_camera_settings(self):
        """Save camera settings to file"""
        try:
            settings = {
                'camera_index': self.camera_index
            }
            with open('camera_settings.json', 'w') as f:
                json.dump(settings, f, indent=2)
            print(f"Saved camera settings")
        except Exception as e:
            print(f"Error saving camera settings: {e}")
    
    def load_models(self):
        """Load the models using Detectron2"""
        print("Loading rebar detection model...")
        
        # Set device explicitly to CPU
        self.device = "cpu"
        
        # Rebar detection model
        self.rebar_cfg = get_cfg()
        self.rebar_cfg.merge_from_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")
        self.rebar_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        self.rebar_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.rebar_cfg.MODEL.DEVICE = "cpu"
        self.rebar_model = build_model(self.rebar_cfg)
        DetectionCheckpointer(self.rebar_model).load("rebar_model1.pth")
        self.rebar_model.eval()
        
        print("Loading section detection model...")
        
        # Section detection model
        self.section_cfg = get_cfg()
        self.section_cfg.merge_from_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")
        self.section_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        self.section_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.section_cfg.MODEL.DEVICE = "cpu"
        self.section_model = build_model(self.section_cfg)
        DetectionCheckpointer(self.section_model).load("section_model1.pth") 
        self.section_model.eval()
    
    def load_cement_ratios(self):
        """Load cement mixture ratios based on rebar diameter"""
        # Default ratios if file doesn't exist
        self.cement_ratios = {
            "small": {"cement": 1, "sand": 2, "aggregate": 3, "diameter_range": [6, 12]},
            "medium": {"cement": 1, "sand": 2, "aggregate": 4, "diameter_range": [12, 20]},
            "large": {"cement": 1, "sand": 3, "aggregate": 5, "diameter_range": [20, 50]}
        }
        
        # Try to load from file
        try:
            if os.path.exists('cement_ratios.json'):
                with open('cement_ratios.json', 'r') as f:
                    self.cement_ratios = json.load(f)
        except Exception as e:
            print(f"Error loading cement ratios: {e}")
    
    def initialize_camera(self):
        """Initialize external USB camera"""
        global latest_frame
        
        try:
            # Clean up existing camera instance if it exists
            if hasattr(self, 'camera') and self.camera is not None:
                try:
                    self.camera.release()
                except:
                    pass
            
            # Initialize OpenCV camera
            self.camera = cv2.VideoCapture(self.camera_index)
            
            # Set camera resolution
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Check if camera opened successfully
            if not self.camera.isOpened():
                raise Exception(f"Could not open camera with index {self.camera_index}")
            
            # Read a test frame
            ret, frame = self.camera.read()
            if not ret:
                raise Exception("Could not read frame from camera")
                
            # Store the frame
            with frame_lock:
                latest_frame = frame.copy()
            
            print(f"Camera initialized with index {self.camera_index}")
            return True
            
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def restart_camera(self):
        """Attempt to restart the camera if there's an issue"""
        try:
            print("Attempting to restart camera...")
            
            # Clean up existing camera if possible
            if hasattr(self, 'camera') and self.camera is not None:
                try:
                    self.camera.release()
                    self.camera = None
                except:
                    pass
            
            # Reinitialize the camera
            time.sleep(1)  # Wait a bit before restarting
            success = self.initialize_camera()
            
            if success:
                print("Camera restarted successfully")
                return True
            else:
                print("Failed to restart camera")
                return False
                
        except Exception as e:
            print(f"Failed to restart camera: {e}")
            return False
    
    def capture_image(self):
        """Capture an image from the camera and analyze it"""
        global latest_frame, latest_distance
        
        # Check if already processing
        if self.is_processing:
            print("Already processing, ignoring capture request")
            return {"success": False, "message": "Already processing an image"}
            
        # Set processing flag
        self.is_processing = True
        
        try:
            # Reset current results data
            self.current_results = []
            
            # Ensure camera is initialized
            if self.camera is None or not self.camera.isOpened():
                if not self.initialize_camera():
                    raise Exception("Failed to initialize camera")
            
            # Use the latest frame from the camera thread
            with frame_lock:
                if latest_frame is None:
                    raise Exception("No frame available")
                self.captured_frame = latest_frame.copy()
            
            # Get the current distance
            with distance_lock:
                current_distance = latest_distance
            
            # Generate timestamp for this analysis session
            self.current_timestamp = time.strftime("%Y%m%d-%H%M%S")
            
            # Create a unique folder for this analysis session
            self.current_result_dir = os.path.join(self.results_dir, f"analysis_{self.current_timestamp}")
            os.makedirs(self.current_result_dir, exist_ok=True)
            
            # Save original image in the analysis folder
            original_filename = os.path.join(self.current_result_dir, 'original_image.jpg')
            cv2.imwrite(original_filename, self.captured_frame)
            
            print("Image captured. Starting analysis...")
            
            # Start the analysis
            result = self.detect_rebar(self.captured_frame)
            
            # Update the distance in the latest_analysis
            latest_analysis["distance"] = current_distance
            
            return result
            
        except Exception as e:
            print(f"Error capturing image: {e}")
            print(traceback.format_exc())
            self.is_processing = False  # Reset processing flag
            
            # Try to restart the camera if there was an error
            self.restart_camera()
            
            return {"success": False, "message": f"Error: {str(e)}"}
        
        finally:
            self.is_processing = False  # Reset processing flag
            # Force garbage collection
            gc.collect()
    
    def detect_rebar(self, frame):
        """First detect if there is a rebar in the image"""
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize for faster processing
        height, width = frame_rgb.shape[:2]
        max_dim = 800
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
            print(f"Resized image for analysis: {width}x{height} -> {new_width}x{new_height}")
        
        # Preprocess for model
        height, width = frame_rgb.shape[:2]
        image = torch.as_tensor(frame_rgb.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image, "height": height, "width": width}
        
        # Run rebar detection model
        with torch.no_grad():
            outputs = self.rebar_model([inputs])[0]
        
        # Check if any rebars were detected
        if len(outputs["instances"]) == 0:
            print("No rebar detected in the image!")
            
            no_rebar_filename = os.path.join(self.current_result_dir, 'no_rebar_detected.jpg')
            cv2.imwrite(no_rebar_filename, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            
            return {"success": False, "message": "No rebar detected in the image"}
        
        # Get the highest-scoring rebar detection
        instances = outputs["instances"].to("cpu")
        scores = instances.scores.numpy()
        boxes = instances.pred_boxes.tensor.numpy()
        
        # Get the best rebar detection
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        best_box = boxes[best_idx].astype(int)
        
        # Draw the detected rebar
        rebar_image = frame_rgb.copy()
        x1, y1, x2, y2 = best_box
        cv2.rectangle(rebar_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(rebar_image, f"Rebar: {best_score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save rebar detection result
        rebar_filename = os.path.join(self.current_result_dir, 'rebar_detected.jpg')
        cv2.imwrite(rebar_filename, cv2.cvtColor(rebar_image, cv2.COLOR_RGB2BGR))
        
        # Now detect sections within the detected rebar region
        return self.detect_sections(frame_rgb, best_box)
    
    def detect_sections(self, frame_rgb, rebar_box):
        """Detect rebar sections within the detected rebar"""
        global latest_distance
        
        # Preprocess for model
        height, width = frame_rgb.shape[:2]
        image = torch.as_tensor(frame_rgb.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image, "height": height, "width": width}
        
        # Run section detection model
        with torch.no_grad():
            outputs = self.section_model([inputs])[0]
        
        # Get the section instances
        instances = outputs["instances"].to("cpu")
        
        if len(instances) == 0:
            print("No rebar sections detected!")
            return {"success": False, "message": "No rebar sections detected!"}
        
        # Get detection details
        scores = instances.scores.numpy()
        boxes = instances.pred_boxes.tensor.numpy()
        masks = instances.pred_masks.numpy() if instances.has("pred_masks") else None
        
        # Create a result image
        result_image = frame_rgb.copy()
        
        # Generate colors for each section
        section_colors = []
        for i in range(len(boxes)):
            color = (
                np.random.randint(0, 200),
                np.random.randint(0, 200),
                np.random.randint(100, 255)
            )
            section_colors.append(color)
        
        # Process each detected section
        print(f"Found {len(boxes)} sections")
        
        # List to store text results for each section
        section_text_results = []
        
        # Get current distance
        with distance_lock:
            current_distance = latest_distance
        
        # Add distance overlay to the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result_image, f"Distance: {current_distance} cm", (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        total_volume = 0
        
        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = map(int, box)
            
            # Calculate diameter in pixels
            width_px = x2 - x1
            height_px = y2 - y1
            diameter_px = min(width_px, height_px)
            
            # Convert to real-world diameter in mm
            mm_per_pixel = 0.1  # Placeholder value
            diameter_mm = diameter_px * mm_per_pixel
            
            # Determine section size based on diameter
            size = "small"
            for category, info in self.cement_ratios.items():
                if "diameter_range" in info:
                    min_diam, max_diam = info["diameter_range"]
                    if min_diam <= diameter_mm < max_diam:
                        size = category
                        break
            
            # Get cement mixture ratio
            ratio = self.cement_ratios[size]
            
            # Calculate volume
            length_cm = height_px * 0.1
            width_cm = width_px * 0.1
            height_cm = width_cm
            volume_cc = length_cm * width_cm * height_cm
            
            # Add to total volume
            total_volume += volume_cc
            
            # Create text result for this section
            section_result = {
                "section_id": i + 1,
                "size_category": size,
                "diameter_mm": round(diameter_mm, 2),
                "confidence": round(score, 3),
                "width_cm": round(width_cm, 2),
                "length_cm": round(length_cm, 2),
                "height_cm": round(height_cm, 2),
                "volume_cc": round(volume_cc, 2),
                "cement_ratio": ratio["cement"],
                "sand_ratio": ratio["sand"],
                "aggregate_ratio": ratio["aggregate"],
                "bbox": [x1, y1, x2, y2]
            }
            section_text_results.append(section_result)
            
            # Save section data for CSV
            section_data = {
                "timestamp": self.current_timestamp,
                **section_result
            }
            self.current_results.append(section_data)
            
            # Draw on the result image
            color = section_colors[i]
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f"S{i+1}"
            cv2.putText(result_image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw mask if available
            if masks is not None:
                mask = masks[i]
                mask_colored = np.zeros_like(frame_rgb)
                mask_colored[mask] = color
                
                # Blend the mask with the image
                alpha = 0.4
                mask_region = mask.astype(bool)
                result_image[mask_region] = (
                    result_image[mask_region] * (1 - alpha) + 
                    mask_colored[mask_region] * alpha
                ).astype(np.uint8)
        
        # Add total volume overlay
        cv2.putText(result_image, f"Volume: {total_volume:.2f} cc", (10, 70), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Save result image
        result_filename = os.path.join(self.current_result_dir, 'section_result.jpg')
        cv2.imwrite(result_filename, cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        
        # Save analysis results to CSV
        self.save_results_to_csv()
        
        # Store the result image for API access
        self.result_image = result_image
        
        # Update API data
        self.update_api_data(total_volume)
        
        return {"success": True, "message": f"Found {len(boxes)} sections, total volume: {total_volume:.2f} cc"}
    
    def save_results_to_csv(self):
        """Save the current analysis results to a CSV file"""
        if not self.current_results:
            print("No results to save")
            return
        
        try:
            # Create a CSV file in the analysis folder
            filename = os.path.join(self.current_result_dir, 'analysis_data.csv')
            
            # Define CSV headers
            headers = [
                "timestamp", "section_id", "size_category", "diameter_mm", 
                "confidence", "width_cm", "length_cm", "height_cm", "volume_cc",
                "cement_ratio", "sand_ratio", "aggregate_ratio"
            ]
            
            # Write data to CSV
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()
                for result in self.current_results:
                    writer.writerow(result)
            
            print(f"Analysis data saved to {filename}")
            
            # Also save a summary text file
            summary_filename = os.path.join(self.current_result_dir, 'summary.txt')
            with open(summary_filename, 'w') as f:
                analysis_time = time.strftime('%Y-%m-%d %H:%M:%S', 
                                            time.localtime(time.mktime(
                                                time.strptime(self.current_timestamp, '%Y%m%d-%H%M%S'))))
                f.write(f"Analysis: {analysis_time}\n\n")
                
                if not self.current_results:
                    f.write("No rebar sections detected.\n")
                else:
                    f.write(f"Found {len(self.current_results)} rebar sections:\n\n")
                    for result in self.current_results:
                        f.write(f"Section {result['section_id']} ({result['size_category']}):\n")
                        f.write(f"  Diameter: {result['diameter_mm']:.1f}mm\n")
                        f.write(f"  Mix: C:{result['cement_ratio']} S:{result['sand_ratio']} A:{result['aggregate_ratio']}\n\n")
            
            print(f"Summary saved to {summary_filename}")
                
        except Exception as e:
            print(f"Error saving analysis data: {e}")
            print(traceback.format_exc())
    
    def update_api_data(self, total_volume):
        """Update the API data after analysis"""
        global latest_analysis
        
        # Only update if we have results
        if not self.current_results:
            return
        
        # Convert the current results to the API format
        segments = []
        
        for result in self.current_results:
            segment = {
                "section_id": result["section_id"],
                "size_category": result.get("size_category", "unknown"),
                "diameter_mm": result.get("diameter_mm", 0),
                "confidence": result.get("confidence", 0.9),
                "width_cm": result.get("width_cm", 0),
                "length_cm": result.get("length_cm", 0),
                "height_cm": result.get("height_cm", 0),
                "volume_cc": result.get("volume_cc", 0),
                "bbox": result.get("bbox", [0, 0, 0, 0])
            }
            segments.append(segment)
        
        # Convert result image to base64 if available
        if self.result_image is not None:
            try:
                pil_img = Image.fromarray(cv2.cvtColor(self.result_image, cv2.COLOR_BGR2RGB))
                img_io = io.BytesIO()
                pil_img.save(img_io, 'JPEG')
                img_io.seek(0)
                img_b64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
                
                latest_analysis["image"] = img_b64
            except Exception as e:
                print(f"Error converting image for API: {e}")
        
        latest_analysis["timestamp"] = self.current_timestamp
        latest_analysis["segments"] = segments
        latest_analysis["total_volume"] = total_volume
        latest_analysis["image_path"] = os.path.join(self.current_result_dir, 'section_result.jpg')
        
        print("API data updated with latest analysis results")

# Function to measure distance using ultrasonic sensor
def measure_distance():
    """Measure distance using ultrasonic sensor"""
    if not USE_DISTANCE_SENSOR:
        return 0.0
        
    try:
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
    except Exception as e:
        print(f"Error measuring distance: {e}")
        return 0.0

# Function to continuously update distance
def distance_monitor():
    """Thread to continuously monitor distance"""
    global latest_distance, camera_running
    
    while camera_running:
        dist = measure_distance()
        if dist > 0:
            with distance_lock:
                latest_distance = dist
                latest_analysis["distance"] = dist
        time.sleep(0.5)  # Update distance every 0.5 seconds

# Function to continuously capture frames
def capture_frames():
    """Thread to continuously capture frames from camera"""
    global latest_frame, camera_running, app_instance
    
    if app_instance is None or not hasattr(app_instance, 'camera'):
        print("Camera not initialized!")
        return
    
    camera = app_instance.camera
    
    print("Starting camera frame capture thread")
    
    while camera_running:
        try:
            # Check if camera is available
            if camera is None or not camera.isOpened():
                app_instance.initialize_camera()
                camera = app_instance.camera
                if camera is None or not camera.isOpened():
                    time.sleep(1)  # Wait before trying again
                    continue
            
            # Read a frame
            ret, frame = camera.read()
            if not ret:
                print("Failed to read frame, trying to reinitialize camera")
                app_instance.initialize_camera()
                camera = app_instance.camera
                continue
            
            # Update the latest frame
            with frame_lock:
                latest_frame = frame.copy()
            
            # Sleep to reduce CPU usage
            time.sleep(0.1)  # 10 fps is adequate for preview
            
        except Exception as e:
            print(f"Error capturing frame: {e}")
            time.sleep(0.5)
    
    # Cleanup when thread exits
    if camera is not None:
        camera.release()
    
    print("Camera frame capture thread stopped")

# Create default cement ratios file
def create_cement_ratios_file():
    """Create default cement ratios file"""
    ratios = {
        "small": {
            "cement": 1, 
            "sand": 2, 
            "aggregate": 3,
            "diameter_range": [6, 12]
        },
        "medium": {
            "cement": 1, 
            "sand": 2, 
            "aggregate": 4,
            "diameter_range": [12, 20]
        },
        "large": {
            "cement": 1, 
            "sand": 3, 
            "aggregate": 5,
            "diameter_range": [20, 50]
        }
    }
    
    with open('cement_ratios.json', 'w') as f:
        json.dump(ratios, f, indent=2)
    
    print("Created default cement ratios file")

# Create default camera settings file
def create_camera_settings_file():
    """Create default camera settings file"""
    settings = {
        "camera_index": 0
    }
    
    with open('camera_settings.json', 'w') as f:
        json.dump(settings, f, indent=2)
    
    print("Created default camera settings file")

# API routes for the Flask server
@api_app.route('/')
def home():
    return "RebarVista API is running!"

@api_app.route('/api/status')
def status():
    """Return the API status"""
    camera_available = False
    if app_instance is not None and hasattr(app_instance, 'camera'):
        if app_instance.camera is not None:
            if hasattr(app_instance.camera, 'isOpened'):
                camera_available = app_instance.camera.isOpened()
            else:
                camera_available = True
    
    return jsonify({
        "status": "online",
        "camera_available": camera_available,
        "has_results": latest_analysis["timestamp"] is not None,
        "distance_sensor_available": USE_DISTANCE_SENSOR
    })

@api_app.route('/api/latest')
def get_latest():
    """Return the latest analysis results (without image)"""
    if latest_analysis["timestamp"] is None:
        return jsonify({
            "timestamp": None,
            "segments": [],
            "total_volume": 0,
            "image_available": False,
            "distance": latest_distance
        })
    
    return jsonify({
        "timestamp": latest_analysis["timestamp"],
        "segments": latest_analysis["segments"],
        "total_volume": latest_analysis["total_volume"],
        "image_available": latest_analysis["image"] is not None,
        "distance": latest_distance
    })

@api_app.route('/api/latest_image')
def get_latest_image():
    """Return the latest analysis image"""
    if latest_analysis["image"] is None:
        return jsonify({"error": "No image available"}), 404
    
    return jsonify({
        "image": latest_analysis["image"]
    })

@api_app.route('/api/distance')
def get_distance():
    """Return the current distance measurement"""
    with distance_lock:
        return jsonify({"distance": latest_distance})

@api_app.route('/api/capture', methods=["POST"])
def trigger_capture():
    """Trigger a new capture and analysis"""
    try:
        # Use global app_instance to trigger a capture
        if app_instance is not None:
            result = app_instance.capture_image()
            return jsonify(result)
        else:
            return jsonify({"success": False, "message": "Application instance not available"}), 500
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@api_app.route('/api/video_feed')
def video_feed():
    """Provide a live video feed"""
    def generate():
        global latest_frame, latest_distance, camera_running
        
        while camera_running:
            # Get the latest frame
            with frame_lock:
                if latest_frame is None:
                    time.sleep(0.1)
                    continue
                    
                frame = latest_frame.copy()
            
            # Add distance overlay
            with distance_lock:
                dist = latest_distance
                
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f"Distance: {dist} cm", (10, 30), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            
            # If analysis results are available, show volume too
            if latest_analysis["total_volume"] > 0:
                cv2.putText(frame, f"Volume: {latest_analysis['total_volume']:.2f} cc", (10, 60), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Encode the frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # Yield the frame in the format expected by Flask
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Brief sleep to reduce CPU usage
            time.sleep(0.1)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@api_app.route('/api/config', methods=["GET"])
def get_config():
    """Return the current configuration"""
    camera_index = 0
    if app_instance is not None:
        threshold = app_instance.rebar_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        if hasattr(app_instance, 'camera_index'):
            camera_index = app_instance.camera_index
    else:
        threshold = 0.7
        
    return jsonify({
        "detection_threshold": threshold,
        "camera_enabled": True,
        "external_camera_index": camera_index,
        "distance_sensor_available": USE_DISTANCE_SENSOR
    })

@api_app.route('/api/config', methods=["POST"])
def update_config():
    """Update the configuration"""
    try:
        config_data = request.json
        
        if app_instance is not None:
            # Update threshold
            if "detection_threshold" in config_data:
                app_instance.rebar_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(config_data["detection_threshold"])
                print(f"Updated detection threshold to {config_data['detection_threshold']}")
            
            # Update camera index
            if "external_camera_index" in config_data:
                camera_index = int(config_data["external_camera_index"])
                if app_instance.camera_index != camera_index:
                    app_instance.camera_index = camera_index
                    app_instance.save_camera_settings()
                    
                    # Force camera reinitialization
                    if hasattr(app_instance, 'camera') and app_instance.camera is not None:
                        app_instance.camera.release()
                        app_instance.camera = None
                    app_instance.initialize_camera()
                    
                    print(f"Updated camera index to {camera_index}")
        
        return jsonify({"message": "Configuration updated"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def cleanup():
    """Clean up resources when the application exits"""
    global camera_running
    
    print("Cleaning up resources...")
    camera_running = False
    
    if app_instance is not None and hasattr(app_instance, 'camera') and app_instance.camera is not None:
        app_instance.camera.release()
    
    if USE_DISTANCE_SENSOR:
        GPIO.cleanup()
    
    print("Cleanup complete")

# Main function
def main():
    global app_instance, camera_running
    
    # Create default files if they don't exist
    if not os.path.exists('cement_ratios.json'):
        create_cement_ratios_file()
    
    if not os.path.exists('camera_settings.json'):
        create_camera_settings_file()
    
    # Set the camera running flag
    camera_running = True
    
    # Create app instance for analysis
    try:
        app_instance = RebarAnalysisApp()
        print("RebarAnalysisApp initialized")
    except Exception as e:
        print(f"Error initializing RebarAnalysisApp: {e}")
        print(traceback.format_exc())
    
    # Start frame capture thread
    frame_thread = threading.Thread(target=capture_frames, daemon=True)
    frame_thread.start()
    
    # Start distance monitoring thread if available
    if USE_DISTANCE_SENSOR:
        distance_thread = threading.Thread(target=distance_monitor, daemon=True)
        distance_thread.start()
    
    try:
        # Start the Flask API server
        api_app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        print("Keyboard interrupt received, shutting down")
    except Exception as e:
        print(f"Error in Flask server: {e}")
        print(traceback.format_exc())
    finally:
        # Clean up resources
        cleanup()

if __name__ == "__main__":
    main()