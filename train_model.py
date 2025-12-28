import cv2
import os
from PIL import Image
import numpy as np



recognizer=cv2.face.LBPHFaceRecognizer_create()
face_cascade=cv2.CascadeClassifier(r"models\haarcascade_frontalface_default.xml")

face_samples=[]
ids=[]

image_no=1
img_paths=[os.path.join("dataset",f) for f in os.listdir("dataset") if f.endswith(".jpg")]
for img_path in img_paths:
    filename=os.path.split(img_path)[-1]
    parts=filename.split(".")
    image_no+=1
    id=int(parts[1])
    pil_img=Image.open(img_path).convert("L")
    img_numpy=np.array(pil_img,'uint8')

    faces=face_cascade.detectMultiScale(img_numpy)
    if len(faces)==0:
        face_samples.append(img_numpy)
        ids.append(id)
    else:
        for x,y,w,h in faces:
            face_samples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
if(ids==0 or face_samples==0 or len(ids)!=len(face_samples)):
    print("not any face recognized.")
    exit()

recognizer.train(face_samples, np.array(ids))
recognizer.write("trainer.yml")
print(f"Model trained and saved as trainer.yml. Trained on {len(set(ids))} users and {len(ids)} images.")



