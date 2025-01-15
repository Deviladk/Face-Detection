import numpy as np
import cv2

face_train=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
video=cv2.VideoCapture(0)
while True:
   
    ret,cam=video.read()
    if not ret:
        print("Fail")
        break
    faces=face_train.detectMultiScale(cam,scaleFactor=1.1,minNeighbors=5)
    for x,y,w,h in faces:
        cam=cv2.rectangle(cam,(x,y),(x+w,y+h),(255,0,0),3)
        cv2.imshow('pro',cam)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

video.release()
cv2.destroyAllWindows()