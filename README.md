# 🖱️ Virtual Mouse Using Hand Gestures

A Python-based **Virtual Mouse** that lets you control your computer mouse using **hand gestures captured through a webcam**.

The project uses **MediaPipe** for real-time hand landmark detection, **OpenCV** for camera and image processing, and **PyAutoGUI** for controlling the system mouse.

With simple hand gestures, users can move the cursor, perform left and right clicks, drag objects, and scroll without using a physical mouse.

---

## ✨ Features

- **Real-Time Hand Tracking**  
  Uses MediaPipe to detect and track hand landmarks through a webcam.

- **Cursor Control**  
  The cursor is controlled using the position of the **index finger**.

- **Left Click**  
  Bring the **thumb close to the index finger**.

- **Right Click**  
  Bring the **thumb close to the middle finger**.

- **Drag and Drop**  
  Hold the **thumb close to the ring finger** to drag objects.

- **Scrolling**  
  Use **thumbs-up** and **thumbs-down** gestures to scroll up and down.

- **Smooth Cursor Movement**  
  Includes cursor sensitivity, deadzone filtering, and adaptive smoothing for more natural movement.

- **Active Control Area**  
  A defined camera area is mapped to the full computer screen for better cursor control.

- **On-Screen Instructions**  
  Gesture instructions are displayed directly on the camera window.

- **FPS Display**  
  Shows the approximate real-time processing FPS.

---

## 🤚 Gesture Controls

| Gesture | Action |
|---|---|
| ☝️ Index Finger Movement | Move Cursor |
| 🤏 Thumb + Index Finger | Left Click |
| 🤏 Thumb + Middle Finger | Right Click |
| 🤏 Thumb + Ring Finger | Drag |
| 👍 Thumb Up | Scroll Up |
| 👎 Thumb Down | Scroll Down |
| `ESC` | Exit Program |

---

## 🛠️ Technologies Used

- **Python**
- **OpenCV**
- **MediaPipe**
- **PyAutoGUI**

---

## 📦 Installation
### User need Python 12 Virtual Environment

### 1. Clone the Repository

```bash
git clone https://github.com/dEVIL8235/Virtual-Mouse.git
cd Virtual-Mouse
```

### 2. Create and Activate a Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Program

Run the following command from the project directory:

```bash
python mouse.py
```

Make sure your **webcam is connected and accessible** before starting the application.

---

## 🎮 How It Works

The application follows this basic process:

```text
Webcam
   ↓
OpenCV Frame Capture
   ↓
MediaPipe Hand Detection
   ↓
21 Hand Landmarks
   ↓
Gesture Recognition
   ↓
Coordinate Mapping
   ↓
Cursor / Mouse Action
   ↓
PyAutoGUI
```

The **index finger landmark** is used for cursor movement, while the relative positions of the thumb and other fingers are used to identify different mouse gestures.

---

## 📸 Result

When the program starts, a webcam window appears showing:

- Live camera feed
- Detected hand landmarks
- Active cursor-control area
- Gesture instructions
- Current FPS

The detected hand movements are translated into corresponding mouse actions in real time.

---

## ⚙️ Configuration

The application's behavior can be adjusted through the configuration values in `mouse.py`.

Examples include:

```python
FRAME_MARGIN = 90
CURSOR_SENSITIVITY = 1.30
DEADZONE = 3

SMOOTHING_FAST = 3
SMOOTHING_MEDIUM = 5
SMOOTHING_SLOW = 8

SCROLL_COOLDOWN = 0.10
SCROLL_AMOUNT = 6
THUMB_SCROLL_THRESHOLD = 50
```

These settings can be tuned according to the webcam setup and desired cursor sensitivity, smoothness, and scrolling speed.

---

## 📁 Project Structure

```text
Virtual-Mouse/
│
├── mouse.py
├── requirements.txt
├── LICENSE.md
└── README.md
```

---

## ⚠️ Notes

- Use the application in a **well-lit environment** for better hand tracking.
- Keep your hand clearly visible to the webcam.
- Cursor accuracy depends on webcam quality, lighting, and hand position.
- The application is designed for **one-hand gesture control**.

---

## 📝 License

This project is licensed under the **MIT License**.

See the [LICENSE.md](LICENSE.md) file for the complete license terms.

---

## 👨‍💻 Author

**Prashant Kumar**

GitHub: [@dEVIL8235](https://github.com/dEVIL8235)

---

## ⭐ Support

If you find this project useful, consider giving the repository a **star ⭐** on GitHub.