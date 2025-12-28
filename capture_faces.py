import cv2
import pandas as pd
import numpy as np
import time
import os

face_cascade=cv2.CascadeClassifier(r"models\haarcascade_frontalface_default.xml")

cap=cv2.VideoCapture(0)

count=0
image_per_person=30

user_id=input("Enter numeric user ID (e.g. 1): ").strip()
user_name=input("Enter user name (e.g. Rahul): ").strip()
user_df=pd.DataFrame(columns=["id","name"])


if os.path.exists("users.csv"):
    user_df = pd.read_csv("users.csv")
else:
    user_df = pd.DataFrame(columns=["id", "name"])


if int(user_id) not in user_df["id"].astype(int).values:
    new_row = {"id": int(user_id), "name": user_name}
    user_df = pd.concat([user_df, pd.DataFrame([new_row])], ignore_index=True)

user_df.to_csv("users.csv", index=False)


while True:
    ret,frame=cap.read()
    if not ret:
        print("can't get video.")
        break
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    
    for x,y,w,h in faces:
        count+=1
        face_img=gray[y:y+h,x:x+w]
        cv2.imwrite(fr"dataset\user.{user_id}.{count}.jpg",face_img)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,f"image{count}/{image_per_person}",(x,y-10),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,0),3)
        time.sleep(0.1)
    cv2.imshow("captured face",frame)

    k=cv2.waitKey(1) & 0xFF
    if k==ord('q') or count>=image_per_person:
        break

print(f"captured{count} images for id={user_id} , name={user_name}")
cap.release()
cv2.destroyAllWindows()

