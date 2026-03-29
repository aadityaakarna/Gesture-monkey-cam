# 🐒 Gesture Monkey Cam

A real-time hand gesture recognition app that pops up monkey reaction images based on what gesture you make in front of your webcam. Built with Python, OpenCV, and Google MediaPipe.

---

## 🎯 How It Works

The app uses your webcam to detect your face and hand positions in real time. No image is shown by default — only when you make one of the three gestures below does the matching monkey image appear beside your face.

| Gesture | How to do it | Image shown |
|---|---|---|
| ☝️ **Index finger up** | Point one finger up, curl the rest | 😄 Smile monkey |
| ✋ **Hand near mouth/chin** | Bring your hand close to your chin | 🐒 Thinking monkey |
| 🤲 **Hand near chest/heart** | Place your palm on your chest | 😮 Surprised monkey |

---

## 📁 Project Structure

```
face_project/
├── vision_test.py                        # Main application
├── open.png                              # Thinking monkey image
├── smile.png                             # Smile monkey image
├── surprise.png                          # Surprised monkey image
├── hand_landmarker.task                  # Auto-downloaded on first run
├── face_landmarker.task                  # Auto-downloaded on first run
├── diagnose_env.py                       # Environment diagnostic script
└── README.md
```

> **Note:** The two `.task` model files are downloaded automatically the first time you run the app. They are large binary files and should be added to `.gitignore`.

---

## ⚙️ Requirements

- Python 3.8 or higher
- A working webcam
- The following Python packages:

```
opencv-python
mediapipe
numpy
```

---

## 🚀 Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS / Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install opencv-python mediapipe numpy
```

### 4. Run the app

```bash
python vision_test.py
```

The MediaPipe model files (`hand_landmarker.task` and `face_landmarker.task`) will be **downloaded automatically** on first run. This only happens once.

---

## 🎮 Usage

1. Run the script and allow webcam access
2. Make sure your face is visible in the frame
3. Try each gesture:
   - Point **one finger up** → smile monkey appears
   - Bring your **hand to your chin** → thinking monkey appears
   - Place your **hand on your chest** → surprised monkey appears
4. Press **ESC** to quit

The status bar at the bottom of the window shows which gesture is currently being detected.

---

## 🔧 Troubleshooting

**Webcam not opening**
- Make sure no other app is using the camera
- Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in `vision_test.py`

**Models not downloading**
- Check your internet connection
- You can manually download them from [MediaPipe Models](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker#models) and place them in the project folder

**Gesture not triggering**
- Make sure your face is detected first (purple dots visible on face)
- For chest gesture, make sure your chest is visible in frame below your face
- Try adjusting your distance from the camera

**Run the diagnostic script**
```bash
python diagnose_env.py
```

---

## 🛠️ Tech Stack

- **[OpenCV](https://opencv.org/)** — webcam capture and image rendering
- **[MediaPipe](https://developers.google.com/mediapipe)** — hand and face landmark detection
- **[NumPy](https://numpy.org/)** — numerical operations for gesture geometry

---

## 📄 License

MIT License — feel free to use and modify this project.