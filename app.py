from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Define color ranges in HSV
COLOR_RANGES = {
    "Red": [(0, 120, 70), (10, 255, 255)],  # Red (lower bound)
    "Blue": [(90, 50, 50), (130, 255, 255)],  # Blue
    "Green": [(36, 50, 50), (86, 255, 255)],  # Green
    "Yellow": [(20, 100, 100), (30, 255, 255)],  # Yellow
    "Orange": [(10, 100, 100), (20, 255, 255)],  # Orange
    "Purple": [(130, 50, 50), (160, 255, 255)],  # Purple
    "Pink": [(160, 50, 50), (180, 255, 255)],  # Pink
    "Cyan": [(85, 50, 50), (100, 255, 255)],  # Cyan
    "White": [(0, 0, 200), (180, 30, 255)],  # White
    "Black": [(0, 0, 0), (180, 255, 30)]  # Black
}

def detect_colors(frame):
    """Detects multiple colors in the frame and overlays text."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for color_name, (lower, upper) in COLOR_RANGES.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:  # Ignore small detections
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
                cv2.putText(frame, color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame

def generate_frames():
    """Yields processed frames for live streaming."""
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = detect_colors(frame)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
