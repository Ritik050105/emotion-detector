import cv2
import mediapipe as mp
from deepface import DeepFace
import os

# Mediapipe setup for face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Directory to save dataset images
dataset_dir = "stress_dataset"
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

def detect_emotion(frame):
    """Detect emotion using DeepFace and return the dominant emotion."""
    try:
        # Perform emotion detection using DeepFace
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']  # Get the dominant emotion
        return emotion
    except Exception as e:
        print("Error in emotion detection:", e)
        return "neutral"

# Real-time face detection
cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Camera not detected.")
        break
    
    # Flip the frame for a mirror view
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect face mesh
    result = face_mesh.process(rgb_frame)
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # Detect emotion using DeepFace
    emotion = detect_emotion(frame)
    
    # Display the detected emotion
    cv2.putText(frame, f"Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Stress Detection", frame)
    
    # Key press events
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break

cap.release()
cv2.destroyAllWindows()
