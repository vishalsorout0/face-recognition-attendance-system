# Face Recognition Attendance System

A real-time face recognition based attendance system using Python and OpenCV.  
This application captures face images, trains a recognition model, and automatically marks attendance with date and time using a webcam.

---

## 📌 Features

- Real-time face detection using Haar Cascade
- Face recognition using LBPH (Local Binary Pattern Histogram)
- Automatic attendance marking with timestamp
- Prevents duplicate attendance for the same person on the same day
- Stores attendance records in CSV format
- Simple, lightweight, and works offline

---

## 🛠 Tech Stack

- Python 3
- OpenCV (opencv-contrib-python)
- NumPy
- Pandas
- Haar Cascade Classifier
- CSV file storage

---

## 📁 Project Structure
```bash
face-recognition-attendance-system/
│
├── dataset/ # Captured face images
├── models/ # opencv model 
├── capture_faces.py # Capture face images
├── train_model.py # Train LBPH model
├── recognize_attendance.py # Recognize faces and mark attendance
├── users.csv # Maps user IDs to names
├── trainer.yml # Trained model file
├── attendance.csv # Attendance records
└── README.md # Project documentation

```
---

## ⚙️ Installation

### 1. Install Python  
Download from: https://www.python.org

### 2. Install dependencies

```bash
pip install opencv-contrib-python numpy pandas pillow
```

---

## 🚀How to Run


### Step 1 — Capture Face Data
```bash

python capture_faces.py

```

-Enter user ID and name.

-Webcam will open.

-30 images will be captured automatically.

### Step 2 — Train the Model

```bash
python train_model.py
```
-Reads images from dataset/

-Trains the LBPH model

-Saves model as trainer.yml

### Step 3 — Recognize Faces & Mark Attendance
```bash
python recognize_attendance.py
```

-Opens webcam

-Detects and recognizes faces

-Marks attendance in attendance.csv

---

## 📊 Attendance Format

Attendance is stored in attendance.csv as:

id	name	datetime
1	Rahul	2025-01-10 09:12:33


---

## 🧠 How It Works

1. Haar Cascade detects faces in video frames.

2. LBPH extracts facial features.

3. Recognizer predicts the closest match.

4. If confidence is acceptable, attendance is recorded.

---

## 👨‍💻 Author

VISHAL SOROUT
GitHub: https://github.com/vishalsorout0

---

## 📜 License

**This project is open-source and free to use for educational purposes.**