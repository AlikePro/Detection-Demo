import cv2
import numpy as np
import mediapipe as mp
import time
import random
from ultralytics import YOLO

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)
model = YOLO("yolov8n.pt")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

cheating_probability = 0
last_head_positions = None
last_eye_positions = None
stable_gaze_start = None
phone_detected_time = 0
def detect_cheating(frame):
    global cheating_probability, last_head_positions, last_eye_positions, stable_gaze_start, phone_detected_time
    height, width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    head_movement = False
    eye_movement = False
    gaze_direction = "STRAIGHT"
    stable_gaze = False
    face_rect = None
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        left_eye = face_landmarks.landmark[33]
        right_eye = face_landmarks.landmark[263]
        nose = face_landmarks.landmark[1]
        chin = face_landmarks.landmark[152]
        forehead = face_landmarks.landmark[10]
        head_tilt = (nose.y - chin.y) * height
        forehead_to_nose = (forehead.y - nose.y) * height
        side_movement = abs(nose.x - 0.5) * width

        if head_tilt > 35 and forehead_to_nose > 15:
            gaze_direction = "DOWN"
        elif side_movement > 60:
            gaze_direction = "SIDE"
        else:
            gaze_direction = "STRAIGHT"
        current_head_position = (nose.x, nose.y)
        if last_head_positions is not None:
            head_distance = np.linalg.norm(np.array(last_head_positions) - np.array(current_head_position))
            if head_distance > 0.005:
                head_movement = True
        last_head_positions = current_head_position

        current_eye_positions = (left_eye.x, left_eye.y, right_eye.x, right_eye.y)
        if last_eye_positions is not None:
            eye_distance = np.linalg.norm(np.array(last_eye_positions) - np.array(current_eye_positions))
            if eye_distance > 0.01:
                eye_movement = True
        last_eye_positions = current_eye_positions

        if gaze_direction in ["DOWN", "SIDE"]:
            if stable_gaze_start is None:
                stable_gaze_start = time.time()
            elif time.time() - stable_gaze_start > 3:
                stable_gaze = True
        else:
            stable_gaze_start = time.time()
        x_min, y_min = int(nose.x * width - 120), int(nose.y * height - 170)
        x_max, y_max = int(nose.x * width + 120), int(nose.y * height + 170)
        face_rect = (x_min, y_min, x_max, y_max)

    results = model(frame)
    phone_detected = False
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 67:
                phone_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "PHONE DETECTED", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    fluctuation = random.uniform(-2, 2)
    if phone_detected:
        cheating_probability += random.uniform(5, 10)
    if stable_gaze:
        cheating_probability += random.uniform(4, 7)
    if head_movement:
        cheating_probability += random.uniform(2, 5)
    if eye_movement:
        cheating_probability += random.uniform(1, 3)
    if not (phone_detected or stable_gaze or head_movement or eye_movement):
        cheating_probability = max(0, cheating_probability - random.uniform(2, 5))
    cheating_probability = max(0, min(100, cheating_probability))
    if face_rect:
        x_min, y_min, x_max, y_max = face_rect
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.putText(frame, f"{cheating_probability:.1f}%", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    return frame

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = detect_cheating(frame)
    cv2.imshow("AI Anti-Cheating", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
