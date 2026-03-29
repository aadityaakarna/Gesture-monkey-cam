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
FACE_MODEL_URL_BLEND = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker_with_blendshapes/float16/1/face_landmarker_with_blendshapes.task"
FACE_MODEL_URL_BASIC = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

HAND_MODEL_PATH = rel("hand_landmarker.task")
FACE_MODEL_PATH_BLEND = rel("face_landmarker_with_blendshapes.task")
FACE_MODEL_PATH_BASIC = rel("face_landmarker.task")

THINKING_IMG_PATH  = rel("open.png")      # thinking
SMILE_IMG_PATH     = rel("smile.png")     # smile
SURPRISED_IMG_PATH = rel("surprise.png")  # surprised

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
HAND_MIDPOINT_RADIUS = 7
HAND_MIDPOINT_RING_COLOR = (255, 255, 255)
HAND_MIDPOINT_RING_THICKNESS = 2

def load_image_bgra(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None: return None
    if img.ndim == 3 and img.shape[2] == 3:
        b,g,r = cv2.split(img)
        a = np.full_like(b, 255)
        img = cv2.merge([b,g,r,a])
    return img

def overlay_bgra(dst, src_bgra, x, y, scale=1.0):
    if src_bgra is None: return
    h, w = src_bgra.shape[:2]
    nw, nh = int(w*scale), int(h*scale)
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

    b,g,r,a = cv2.split(roi_src)
    a = (a.astype(np.float32) / 255.0)
    a3 = cv2.merge([a,a,a])
    fg = cv2.merge([b,g,r]).astype(np.float32)
    bg = roi_dst.astype(np.float32)
    blended = a3*fg + (1.0 - a3)*bg
    dst[y1:y2, x1:x2] = blended.astype(np.uint8)

# ---------- Face expression (mouth) ----------
def get_bs_score(bs, name):
    try:
        return next((c.score for c in bs.categories if c.category_name == name), 0.0)
    except Exception:
        return 0.0

def mouth_geometry(face_lms):
    def P(idx):
        lm = face_lms[idx]
        return np.array([lm.x, lm.y], dtype=np.float32)
    left, right = P(61), P(291)
    up, down   = P(13), P(14)
    inner_l, inner_r = P(78), P(308)
    mouth_open  = np.linalg.norm(down - up)
    mouth_width = np.linalg.norm(right - left)
    inner_width = np.linalg.norm(inner_r - inner_l)
    return mouth_open, mouth_width, inner_width

# MOUTH CLASSIFIER: make smile win over surprised when teeth/cheeks indicate smiling; gate surprised by low smile.
def classify_expression_blendshapes(bs):
    if bs is None or not hasattr(bs, "categories"):
        return None, (0,0,0)
    open_score   = max(get_bs_score(bs, "mouthOpen"), get_bs_score(bs, "jawOpen"))
    smile_score  = (get_bs_score(bs, "mouthSmileLeft") + get_bs_score(bs, "mouthSmileRight")) / 2.0
    pucker_score = max(get_bs_score(bs, "mouthPucker"), get_bs_score(bs, "mouthFunnel"))

    # Prioritize smile when smiling is strong, even if mouth is somewhat open (teeth showing)
    if smile_score >= 0.55 and pucker_score < 0.50:
        return "smile", (open_score, smile_score, pucker_score)
    # Surprised only when mouth is very open AND smiling is weak
    if open_score >= 0.75 and smile_score < 0.45 and pucker_score < 0.55:
        return "surprised", (open_score, smile_score, pucker_score)
    # Thinking requires strong pucker and low open/smile
    if pucker_score >= 0.65 and open_score < 0.50 and smile_score < 0.50:
        return "thinking", (open_score, smile_score, pucker_score)
    return None, (open_score, smile_score, pucker_score)

def classify_expression_geom(face_lms):
    if not face_lms: return None, (0,0,0)
    mo, mw, iw = mouth_geometry(face_lms)
    # Smile wins if corners are wide even if mouth is a bit open
    if mw > 0.45 and mo < 0.08 and iw > 0.14:
        return "smile", (mo, mw, iw)
    # Surprised only if mouth is very open and smile width is not huge
    if mo > 0.09 and iw > 0.11 and mw < 0.48:
        return "surprised", (mo, mw, iw)
    # Thinking if inner width is small (pucker) and mouth isn't open
    if iw < 0.10 and mo < 0.07 and mw < 0.40:
        return "thinking", (mo, mw, iw)
    return None, (mo, mw, iw)

# ---------- Hand gesture detection ----------
def angle_deg(a, b, c):
    v1 = np.array(a) - np.array(b)
    v2 = np.array(c) - np.array(b)
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6: return 180.0
    cosang = np.clip(np.dot(v1, v2) / (n1*n2), -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def finger_extended(hand, w, h, mcp, pip, dip, tip):
    p_mcp = landmark_to_point(hand[mcp], w, h)
    p_pip = landmark_to_point(hand[pip], w, h)
    p_tip = landmark_to_point(hand[tip], w, h)
    ang = angle_deg(p_mcp, p_pip, p_tip)
    dist_tip_mcp = np.linalg.norm(np.array(p_tip) - np.array(p_mcp))
    dist_pip_mcp = np.linalg.norm(np.array(p_pip) - np.array(p_mcp))
    return ang < 32 and dist_tip_mcp > dist_pip_mcp + 16  # stricter to avoid false positives

def index_up(hand, w, h):
    idx_ext  = finger_extended(hand, w, h, 5, 6, 7, 8)
    mid_ext  = finger_extended(hand, w, h, 9,10,11,12)
    ring_ext = finger_extended(hand, w, h,13,14,15,16)
    pink_ext = finger_extended(hand, w, h,17,18,19,20)
    return idx_ext and not (mid_ext or ring_ext or pink_ext)

def palm_center(hand, w, h):
    pts = [landmark_to_point(hand[i], w, h) for i in range(0, 21)]
    arr = np.array(pts, dtype=np.float32)
    return np.mean(arr, axis=0)

def average_curl(hand, w, h):
    def curl(mcp,pip,tip):
        p_mcp = landmark_to_point(hand[mcp], w, h)
        p_pip = landmark_to_point(hand[pip], w, h)
        p_tip = landmark_to_point(hand[tip], w, h)
        return angle_deg(p_mcp, p_pip, p_tip)
    curls = [curl(5,6,8), curl(9,10,12), curl(13,14,16), curl(17,18,20)]
    return float(np.mean(curls))

def hands_clasped(hands, w, h):
    if len(hands) < 2: return False
    c0 = palm_center(hands[0], w, h)
    c1 = palm_center(hands[1], w, h)
    dist = np.linalg.norm(c0 - c1)
    curl0 = average_curl(hands[0], w, h)
    curl1 = average_curl(hands[1], w, h)
    return dist < 0.11 * w and (curl0 > 60 and curl1 > 60)

# HAND NEAR MOUTH: per-face check — if index fingertip or thumb tip is close to mouth, map to thinking
def hand_near_mouth(face_lms, hands, w, h):
    if not hands: return False
    # Mouth center from inner lip landmarks
    mx_n = (face_lms[13].x + face_lms[14].x) / 2.0
    my_n = (face_lms[13].y + face_lms[14].y) / 2.0
    mx, my = int(mx_n * w), int(my_n * h)
    # Radius proportional to face width
    xs = [lm.x for lm in face_lms]; ys = [lm.y for lm in face_lms]
    x_min, x_max = min(xs), max(xs)
    face_w_px = max(1, int((x_max - x_min) * w))
    radius = int(0.10 * face_w_px)  # ~10% of face width
    tips = [8, 4]  # index tip, thumb tip
    for hand in hands:
        for t in tips:
            tx, ty = landmark_to_point(hand[t], w, h)
            if (tx - mx)**2 + (ty - my)**2 <= radius**2:
                return True
    return False

# ---------- Smoothing ----------
FACE_EXPR_STATE = {}  # key -> counters per expr
def face_key(face_box):
    x_min, y_min, x_max, y_max = face_box
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    return (int(cx // 40), int(cy // 40))

def update_state(key, expr):
    st = FACE_EXPR_STATE.setdefault(key, {"smile":0, "surprised":0, "thinking":0})
    for k in st: st[k] = max(0, st[k] - 1)     # decay
    if expr in st: st[expr] = min(15, st[expr] + 4)  # boost current
    best = max(st, key=lambda k: st[k])
    return best if st[best] >= 6 else None

def place_to_side(face_box, overlay_w, overlay_h, frame_w, frame_h, margin=10):
    x_min, y_min, x_max, y_max = face_box
    x = x_max + margin
    y = y_min
    if x + overlay_w > frame_w:
        x = x_min - overlay_w - margin
    y = max(0, min(y, frame_h - overlay_h))
    return x, y

def main():
    # Ensure models
    ensure_task(HAND_MODEL_PATH, HAND_MODEL_URL, "Hand model")
    using_blendshapes = True
    try:
        ensure_task(FACE_MODEL_PATH_BLEND, FACE_MODEL_URL_BLEND, "Face model (with blendshapes)")
        face_model_path = FACE_MODEL_PATH_BLEND
    except Exception as e:
        print(f"[WARN] Blendshapes unusable ({e}); falling back to basic model.")
        ensure_task(FACE_MODEL_PATH_BASIC, FACE_MODEL_URL_BASIC, "Basic face model")
        face_model_path = FACE_MODEL_PATH_BASIC
        using_blendshapes = False

    # Create landmarkers
    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    face_options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=face_model_path),
        output_face_blendshapes=using_blendshapes,
        min_face_detection_confidence=0.6,
        num_faces=5,
    )
    print(f"Using face model: {face_model_path} (blendshapes={using_blendshapes})")

    hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
    face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

    # Load overlays
    overlays = {
        "thinking": load_image_bgra(THINKING_IMG_PATH),
        "smile":    load_image_bgra(SMILE_IMG_PATH),
        "surprised":load_image_bgra(SURPRISED_IMG_PATH),
    }

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    # Auto calibration for mouth (first ~60 frames)
    auto_ready = False
    blend_buf = deque(maxlen=60)
    geom_buf  = deque(maxlen=60)
    bases = {"open":0.0, "smile":0.0, "pucker":0.0, "mo":0.03, "mw":0.30, "iw":0.12}

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Detect
            hand_result = hand_landmarker.detect(mp_image)
            face_result = face_landmarker.detect(mp_image)

            # Draw hands
            if hand_result.hand_landmarks:
                for hand_lms in hand_result.hand_landmarks:
                    for (s,e) in HAND_CONNECTIONS:
                        p1, p2 = hand_lms[s], hand_lms[e]
                        x1,y1 = landmark_to_point(p1,w,h)
                        x2,y2 = landmark_to_point(p2,w,h)
                        cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)
                        mx,my = (x1+x2)//2, (y1+y2)//2
                        cv2.circle(frame,(mx,my),HAND_MIDPOINT_RADIUS,HAND_MIDPOINT_RING_COLOR,HAND_MIDPOINT_RING_THICKNESS)
                    for lm in hand_lms:
                        x,y = landmark_to_point(lm,w,h)
                        cv2.circle(frame,(x,y),3,(0,255,0),-1)

            # Faces + overlays
            num_faces = 0
            if face_result.face_landmarks:
                for i, face_lms in enumerate(face_result.face_landmarks):
                    for lm in face_lms:
                        x,y = landmark_to_point(lm,w,h)
                        cv2.circle(frame,(x,y),1,(255,0,0),-1)

                    xs = [int(lm.x*w) for lm in face_lms]
                    ys = [int(lm.y*h) for lm in face_lms]
                    x_min, x_max = max(0,min(xs)), min(w-1,max(xs))
                    y_min, y_max = max(0,min(ys)), min(h-1,max(ys))
                    face_box = (x_min,y_min,x_max,y_max)

                    # Mouth features and auto-baseline for first ~60 frames
                    if using_blendshapes and getattr(face_result, "face_blendshapes", None):
                        bs = face_result.face_blendshapes[i]
                        open_score   = max(get_bs_score(bs, "mouthOpen"), get_bs_score(bs, "jawOpen"))
                        smile_score  = (get_bs_score(bs, "mouthSmileLeft") + get_bs_score(bs, "mouthSmileRight")) / 2.0
                        pucker_score = max(get_bs_score(bs, "mouthPucker"), get_bs_score(bs, "mouthFunnel"))
                        if not auto_ready:
                            blend_buf.append((open_score, smile_score, pucker_score))
                            if len(blend_buf) >= blend_buf.maxlen:
                                bases["open"]   = float(np.median([b[0] for b in blend_buf]))
                                bases["smile"]  = float(np.median([b[1] for b in blend_buf]))
                                bases["pucker"] = float(np.median([b[2] for b in blend_buf]))
                                auto_ready = True
                                print(f"[AutoCalib] open={bases['open']:.2f} smile={bases['smile']:.2f} pucker={bases['pucker']:.2f}")
                        # Apply margins to avoid misclassification (relative to baseline)
                        open_adj   = open_score   - bases["open"]
                        smile_adj  = smile_score  - bases["smile"]
                        pucker_adj = pucker_score - bases["pucker"]
                        face_expr_raw, dbg_vals = classify_expression_blendshapes(bs)
                        dbg = f"open:{open_score:.2f} smile:{smile_score:.2f} pucker:{pucker_score:.2f}" + ("" if auto_ready else " [calibrating]")
                    else:
                        mo, mw, iw = mouth_geometry(face_lms)
                        if not auto_ready:
                            geom_buf.append((mo,mw,iw))
                            if len(geom_buf) >= geom_buf.maxlen:
                                bases["mo"] = float(np.median([g[0] for g in geom_buf]))
                                bases["mw"] = float(np.median([g[1] for g in geom_buf]))
                                bases["iw"] = float(np.median([g[2] for g in geom_buf]))
                                auto_ready = True
                                print(f"[AutoCalib] mo={bases['mo']:.3f} mw={bases['mw']:.3f} iw={bases['iw']:.3f}")
                        face_expr_raw, dbg_vals = classify_expression_geom(face_lms)
                        dbg = f"mo:{mo:.3f} mw:{mw:.3f} iw:{iw:.3f}" + ("" if auto_ready else " [calibrating]")

                    cv2.putText(frame, dbg, (x_min, max(0,y_min-8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

                    # HAND NEAR MOUTH overrides to thinking if true
                    hands = hand_result.hand_landmarks or []
                    near_mouth = hand_near_mouth(face_lms, hands, w, h)
                    expr_raw = ("thinking" if near_mouth else face_expr_raw)

                    # Temporal smoothing
                    kkey = face_key(face_box)
                    expr = update_state(kkey, expr_raw)

                    # Show overlay only if a real expression has persisted
                    if expr and overlays.get(expr) is not None:
                        ov = overlays[expr]
                        face_w = max(1, x_max - x_min)
                        scale = min(1.6, max(0.3, (0.8 * face_w) / ov.shape[1]))
                        ow, oh = int(ov.shape[1]*scale), int(ov.shape[0]*scale)
                        ox, oy = place_to_side(face_box, ow, oh, w, h, margin=12)
                        overlay_bgra(frame, ov, ox, oy, scale=scale)
                        cv2.putText(frame, expr.title(), (ox, max(0, oy-6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

                    num_faces += 1

            num_hands = len(hand_result.hand_landmarks) if hand_result.hand_landmarks else 0
            cv2.putText(frame, f"Hands: {num_hands}  Faces: {num_faces}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)

            cv2.imshow("Hands + Faces + Expression Overlays", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()