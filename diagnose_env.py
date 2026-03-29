import os, sys
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
rel = lambda p: os.path.join(BASE_DIR, p)

print("CWD:", os.getcwd())
print("Script dir:", BASE_DIR)
print("Python:", sys.version)
print("mediapipe:", mp.__version__, "cv2:", cv2.__version__)
print("Has cv2.imshow:", hasattr(cv2, "imshow"))

assets = [
    "hand_landmarker.task",
    "face_landmarker_with_blendshapes.task",
    "open.png",
    "smile.png",
    "surprise.png",
]
for p in assets:
    print(f"{p} exists:", os.path.exists(rel(p)))

cap = cv2.VideoCapture(0)
print("Camera opened:", cap.isOpened())
ret, frame = cap.read()
print("First frame:", ret, (None if not ret else frame.shape))

if ret:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    try:
        face_options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=rel("face_landmarker_with_blendshapes.task")),
            output_face_blendshapes=True,
            num_faces=1,
        )
        face_landmarker = vision.FaceLandmarker.create_from_options(face_options)
        res = face_landmarker.detect(mp_image)
        print("Got landmarks:", bool(res.face_landmarks), "blendshapes:", bool(res.face_blendshapes))
    except Exception as e:
        print("Face landmarker error:", repr(e))

cap.release()