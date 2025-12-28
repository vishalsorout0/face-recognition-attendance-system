# recognize_attendance.py
import cv2
import pandas as pd
import os
from datetime import datetime

TRAINER_FILE = "trainer.yml"
USERS_CSV = "users.csv"
ATTENDANCE_CSV = "attendance.csv"
CAM_ID = 0
CONFIDENCE_THRESHOLD = 60  # lower is better; tune if wrong

# Load user mapping
if not os.path.exists(USERS_CSV):
    print("users.csv not found. Run capture_faces.py to create users.")
    exit()
users_df = pd.read_csv(USERS_CSV).set_index("id")["name"].to_dict()

# Prepare recognizer and cascade
if not os.path.exists(TRAINER_FILE):
    print("trainer.yml not found. Run train_model.py first.")
    exit()

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(TRAINER_FILE)

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# Load existing attendance
if os.path.exists(ATTENDANCE_CSV):
    attendance_df = pd.read_csv(ATTENDANCE_CSV)
else:
    attendance_df = pd.DataFrame(columns=["id", "name", "datetime"])

cap = cv2.VideoCapture(CAM_ID)
print("Starting recognition. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from camera")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        id_pred, conf = recognizer.predict(face_img)  # id, confidence
        if conf < CONFIDENCE_THRESHOLD:
            user_name = users_df.get(id_pred, "Unknown")
            label = f"{user_name} ({id_pred})"
            # Avoid duplicate for same day (date only)
            today_date = datetime.now().date()
            already = False
            if not attendance_df.empty:
                # check if same id and same date present
                attendance_df['date_only'] = pd.to_datetime(attendance_df['datetime']).dt.date
                already = ((attendance_df['id'] == id_pred) & (attendance_df['date_only'] == today_date)).any()
                attendance_df = attendance_df.drop(columns=['date_only'])
            if not already:
                nowstr = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                new_row = pd.DataFrame([{
                    "id": int(id_pred),
                    "name": user_name,
                    "datetime": nowstr
                }])

                attendance_df = pd.concat([attendance_df, new_row], ignore_index=True)
                attendance_df.to_csv(ATTENDANCE_CSV, index=False)
                print(f"Marked attendance: {id_pred} - {user_name} at {nowstr}")
        else:
            label = "Unknown"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        cv2.putText(frame, f"Conf:{int(conf)}", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)

    cv2.imshow("Recognize & Attendance", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done. Attendance saved to", ATTENDANCE_CSV)
