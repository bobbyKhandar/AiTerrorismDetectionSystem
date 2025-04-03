import cv2
import os
import time
import random
import string
from ultralytics import YOLO

# Function to generate random filenames
def random_string(length=10):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

# Load YOLO model once (Global Variable)
MODEL_PATH = r"C:/Users/Exam/Desktop/bobby/runs/detect/train13/weights/best.pt"
if not os.path.exists(MODEL_PATH):
    print("Error: YOLO model file not found!")
    exit()
model = YOLO(MODEL_PATH)

# Define the video file path
video_path = r"C:/Users/Exam/Desktop/bobby/gunImagesFromYt/videoplayback.mp4"
cap = cv2.VideoCapture(video_path)

# Check if the video file opens
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get video frame rate
frame_rate = cap.get(cv2.CAP_PROP_FPS) or 30  # Default FPS if undetectable
frame_interval = int(frame_rate / 2)  # Process every 0.5 seconds
lastTime=time.time()
while True:
    currentTime=time.time()
    if currentTime-lastTime >0.1:
        lastTime=currentTime
        ret, frame = cap.read()
        if not ret:
            break  # Exit if no frame is read

    # Run YOLO object detection on the frame
        results = model(frame)

    # Get processed frame with bounding boxes
        processed_frame = results[0].plot()  # YOLO's built-in function to draw bounding boxes

    # Show AI-processed video feed
    cv2.imshow('AI Processed Video Feed', processed_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
