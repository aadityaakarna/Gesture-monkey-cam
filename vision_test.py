#!/usr/bin/env python3
import os, zipfile, urllib.request, math
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def rel(*parts): return os.path.join(BASE_DIR, *parts)

HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

HAND_MODEL_PATH = rel("hand_landmarker.task")
FACE_MODEL_PATH = rel("face_landmarker.task")

OPEN_IMG_PATH      = rel("open.png")      # hand near mouth  -> thinking monkey
SMILE_IMG_PATH     = rel("smile.png")     # index finger up  -> smile monkey
SURPRISED_IMG_PATH = rel("surprise.png")  # hand near chest  -> surprised monkey

def download(url, dst):
    print(f"Downloading: {url}")
    urllib.request.urlretrieve(url, dst)
    print(f"Saved to: {dst}")

def ensure_task(path, url, label):
    if not os.path.exists(path) or not zipfile.is_zipfile(path):
        if os.path.exists(path) and not zipfile.is_zipfile(path):
            print(f"[WARN] {label} at {path} is invalid; re-downloading.")
        download(url, path)
    if not zipfile.is_zipfile(path):
        raise RuntimeError(f"{label} at {path} is not a valid .task ZIP.")

# ---------- Draw helpers ----------
def landmark_to_point(lm, width, height):
    return int(lm.x * width), int(lm.y * height)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

def load_image_bgra(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None: return None
    if img.ndim == 3 and img.shape[2] == 3:
        b, g, r = cv2.split(img)
        a = np.full_like(b, 255)
        img = cv2.merge([b, g, r, a])
    return img

def overlay_bgra(dst, src_bgra, x, y, scale=1.0):
    if src_bgra is None: return
    h, w = src_bgra.shape[:2]
    nw, nh = int(w * scale), int(h * scale)
    if nw <= 0 or nh <= 0: return
    src = cv2.resize(src_bgra, (nw, nh), interpolation=cv2.INTER_AREA)
    H, W = dst.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + nw), min(H, y + nh)
    if x1 >= x2 or y1 >= y2: return
    sx1, sy1 = x1 - x, y1 - y
    sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)
    roi_dst = dst[y1:y2, x1:x2]
    roi_src = src[sy1:sy2, sx1:sx2]
    b, g, r, a = cv2.split(roi_src)
    a = a.astype(np.float32) / 255.0
    a3 = cv2.merge([a, a, a])
    fg = cv2.merge([b, g, r]).astype(np.float32)
    bg = roi_dst.astype(np.float32)
    blended = a3 * fg + (1.0 - a3) * bg
    dst[y1:y2, x1:x2] = blended.astype(np.uint8)

# ---------- Geometry helpers ----------
def angle_deg(a, b, c):
    v1 = np.array(a) - np.array(b)
    v2 = np.array(c) - np.array(b)
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6: return 180.0
    cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def tip_above_mcp(hand, h, tip_idx, mcp_idx, margin=20):
    """True when fingertip Y is clearly above (smaller) its MCP knuckle Y."""
    return (hand[tip_idx].y * h) < (hand[mcp_idx].y * h) - margin

def tip_below_mcp(hand, h, tip_idx, mcp_idx, margin=10):
    """True when fingertip Y is at or below its MCP knuckle Y (curled)."""
    return (hand[tip_idx].y * h) >= (hand[mcp_idx].y * h) - margin

def palm_center(hand, w, h):
    pts = [landmark_to_point(hand[i], w, h) for i in range(21)]
    return np.mean(np.array(pts, dtype=np.float32), axis=0)

# ---------- GESTURE DETECTION ----------

def gesture_index_up(hand, w, h):
    """
    SMILE: Index finger pointing up, all other fingers curled.
    Tip Y < MCP Y means finger is extended upward.
    """
    idx_up   = tip_above_mcp(hand, h, tip_idx=8,  mcp_idx=5,  margin=20)
    mid_down = tip_below_mcp(hand, h, tip_idx=12, mcp_idx=9,  margin=10)
    rng_down = tip_below_mcp(hand, h, tip_idx=16, mcp_idx=13, margin=10)
    pnk_down = tip_below_mcp(hand, h, tip_idx=20, mcp_idx=17, margin=10)
    return idx_up and mid_down and rng_down and pnk_down

def gesture_hand_near_mouth(hand, face_lms, w, h):
    """
    OPEN (thinking): Any fingertip is close to the mouth/chin zone.
    Like the monkey in open.png with hand near chin.
    """
    if face_lms is None: return False
    # Mouth center (upper + lower inner lip)
    mouth_x = int(((face_lms[13].x + face_lms[14].x) / 2.0) * w)
    mouth_y = int(((face_lms[13].y + face_lms[14].y) / 2.0) * h)
    # Chin landmark 152
    chin_x = int(face_lms[152].x * w)
    chin_y = int(face_lms[152].y * h)
    # Target = midpoint between mouth and chin
    target_x = (mouth_x + chin_x) // 2
    target_y = (mouth_y + chin_y) // 2
    # Radius = 30% of face width — generous zone
    xs = [lm.x for lm in face_lms]
    face_w_px = max(1, int((max(xs) - min(xs)) * w))
    radius = int(0.30 * face_w_px)
    # Check all fingertips and palm center
    for tip_idx in [4, 8, 12, 16, 20]:
        tx, ty = landmark_to_point(hand[tip_idx], w, h)
        if (tx - target_x)**2 + (ty - target_y)**2 <= radius**2:
            return True
    return False

def gesture_hand_near_chest(hand, face_lms, w, h):
    """
    SURPRISED: Palm or wrist is in the chest/heart zone below the face.
    Like the monkey in surprise.png with hands on chest.
    """
    if face_lms is None: return False
    xs = [lm.x for lm in face_lms]
    ys = [lm.y for lm in face_lms]
    face_cx    = int(((max(xs) + min(xs)) / 2.0) * w)
    face_bottom = int(max(ys) * h)
    face_h_px  = int((max(ys) - min(ys)) * h)
    face_w_px  = int((max(xs) - min(xs)) * w)
    # Chest zone: horizontally wide, vertically 0.3–2.5 face-heights below face
    chest_y_top    = face_bottom + int(0.3 * face_h_px)
    chest_y_bottom = face_bottom + int(2.5 * face_h_px)
    chest_x_left   = face_cx - int(1.5 * face_w_px)
    chest_x_right  = face_cx + int(1.5 * face_w_px)
    # Check palm center and wrist
    pc = palm_center(hand, w, h)
    wr = np.array(landmark_to_point(hand[0], w, h), dtype=np.float32)
    for pt in [pc, wr]:
        px, py = int(pt[0]), int(pt[1])
        if chest_x_left <= px <= chest_x_right and chest_y_top <= py <= chest_y_bottom:
            return True
    return False

# ---------- Smoothing ----------
GESTURE_BUFFER = deque(maxlen=10)

def smooth_gesture(new_gesture):
    """Return the stable gesture only when it appears in at least half the buffer."""
    GESTURE_BUFFER.append(new_gesture)
    counts = {}
    for g in GESTURE_BUFFER:
        if g is not None:
            counts[g] = counts.get(g, 0) + 1
    if not counts:
        return None
    best = max(counts, key=lambda k: counts[k])
    if counts[best] >= max(3, len(GESTURE_BUFFER) // 2):
        return best
    return None

# ---------- Overlay placement ----------
def place_to_side(face_box, overlay_w, overlay_h, frame_w, frame_h, margin=40):
    x_min, y_min, x_max, y_max = face_box
    x = x_max + margin
    y = y_min
    if x + overlay_w > frame_w:
        x = x_min - overlay_w - margin
    y = max(0, min(y, frame_h - overlay_h))
    return x, y

# ---------- Main ----------
def main():
    ensure_task(HAND_MODEL_PATH, HAND_MODEL_URL, "Hand model")
    ensure_task(FACE_MODEL_PATH, FACE_MODEL_URL, "Face model")

    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    face_options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
        output_face_blendshapes=False,
        min_face_detection_confidence=0.5,
        num_faces=1,
    )

    hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
    face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

    overlays = {
        "open":      load_image_bgra(OPEN_IMG_PATH),
        "smile":     load_image_bgra(SMILE_IMG_PATH),
        "surprised": load_image_bgra(SURPRISED_IMG_PATH),
    }
    for name, img in overlays.items():
        status = "OK" if img is not None else "MISSING - check file path!"
        print(f"  Overlay '{name}': {status}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print("\n=== GESTURE GUIDE ===")
    print("  ☝️  Index finger UP         -> Smile monkey")
    print("  ✋  Hand near MOUTH / CHIN  -> Thinking monkey (open.png)")
    print("  🤲  Hand near CHEST / HEART -> Surprised monkey")
    print("  No gesture                  -> No image shown")
    print("  Press ESC to quit\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            hand_result = hand_landmarker.detect(mp_image)
            face_result = face_landmarker.detect(mp_image)

            # Get face landmarks for position reference (first face only)
            face_lms = None
            face_box = None
            if face_result.face_landmarks:
                face_lms = face_result.face_landmarks[0]
                xs = [int(lm.x * w) for lm in face_lms]
                ys = [int(lm.y * h) for lm in face_lms]
                face_box = (max(0, min(xs)), max(0, min(ys)),
                            min(w - 1, max(xs)), min(h - 1, max(ys)))
                # Draw subtle face dots
                for lm in face_lms:
                    x, y = landmark_to_point(lm, w, h)
                    cv2.circle(frame, (x, y), 1, (180, 100, 255), -1)

            # Draw hands with skeleton
            if hand_result.hand_landmarks:
                for hand_lms in hand_result.hand_landmarks:
                    for (s, e) in HAND_CONNECTIONS:
                        x1, y1 = landmark_to_point(hand_lms[s], w, h)
                        x2, y2 = landmark_to_point(hand_lms[e], w, h)
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    for lm in hand_lms:
                        x, y = landmark_to_point(lm, w, h)
                        cv2.circle(frame, (x, y), 4, (0, 220, 0), -1)

            # ---------- Detect gesture (priority: index_up > near_mouth > near_chest) ----------
            raw_gesture = None
            if hand_result.hand_landmarks:
                for hand_lms in hand_result.hand_landmarks:
                    if gesture_index_up(hand_lms, w, h):
                        raw_gesture = "smile"
                        break
                    elif gesture_hand_near_mouth(hand_lms, face_lms, w, h):
                        raw_gesture = "open"
                        break
                    elif gesture_hand_near_chest(hand_lms, face_lms, w, h):
                        raw_gesture = "surprised"
                        break

            # Smooth over last 10 frames to avoid flickering
            gesture = smooth_gesture(raw_gesture)

            # ---------- Show monkey overlay beside the face ----------
            if gesture and overlays.get(gesture) is not None and face_box is not None:
                ov = overlays[gesture]
                face_w_px = max(1, face_box[2] - face_box[0])
                scale = min(2.5, max(0.7, (1.3 * face_w_px) / ov.shape[1]))
                ow = int(ov.shape[1] * scale)
                oh = int(ov.shape[0] * scale)
                ox, oy = place_to_side(face_box, ow, oh, w, h, margin=40)
                overlay_bgra(frame, ov, ox, oy, scale=scale)

            # ---------- HUD ----------
            num_hands = len(hand_result.hand_landmarks) if hand_result.hand_landmarks else 0
            num_faces = len(face_result.face_landmarks) if face_result.face_landmarks else 0
            cv2.putText(frame, f"Hands:{num_hands}  Faces:{num_faces}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Status bar at bottom
            if gesture:
                labels = {
                    "open":      "THINKING  (hand near mouth)",
                    "smile":     "SMILE  (finger pointing up)",
                    "surprised": "SURPRISED  (hand on chest)",
                }
                cv2.putText(frame, labels.get(gesture, gesture.upper()),
                            (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No gesture detected",
                            (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

            cv2.imshow("Gesture Monkey Cam  [ESC to quit]", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()