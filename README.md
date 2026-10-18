# SentinEye
> AI-powered real-time anti-shoulder surfing and privacy surveillance defense system.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10-green.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.11-orange.svg)](https://mediapipe.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**SentinEye** is a computer vision-based cybersecurity defense tool designed to protect your laptop screen from "shoulder surfers" and unauthorized onlookers in public spaces. Using real-time facial mesh tracking and Perspective-n-Point (PnP) head pose estimation, it detects secondary viewers looking at your display and triggers instant visual alerts.

---

##  Features
- **Real-time Primary & Snooper Detection:** Automatically classifies the primary user (green box) and potential threats (red box with head yaw/pitch angles).
- **Lightweight & Fast:** Runs locally on CPU using MediaPipe and OpenCV without requiring heavy GPU clusters.
- **Privacy First:** No frame or facial data is stored or transmitted; all processing happens locally on your device.

---

##  Installation & Setup

1. **Clone the repository:**
```bash
   git clone https://github.com/FatemeMolaeiJahromi/SentinEye.git
   cd SentinEye

   Create and activate a virtual environment:
bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   
Install dependencies:
bash
   pip install -r requirements.txt
   
 Usage
Run the real-time detection script:

bash
python test_cam.py
Press q to exit the application.

 Project Structure
text
SentinEye/
│
├── core/
│   └── shoulder_surf.py    # Core AI detection & PnP head pose estimation logic
├── test_cam.py             # Real-time webcam execution & alerting UI
├── requirements.txt        # Project dependencies
└── README.md               # Documentation
