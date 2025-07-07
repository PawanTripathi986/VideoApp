import cv2
from script_generator import cartoonize_frame

def generate_live_frames():
    cap = cv2.VideoCapture(0)  # Replace with IP cam URL if needed
    while True:
        success, frame = cap.read()
        if not success:
            break
        cartoon = cartoonize_frame(frame)
        _, buffer = cv2.imencode('.jpg', cartoon)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
