import cv2
from deepface import DeepFace
import warnings
warnings.filterwarnings('ignore')

face_tr = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def det_gen(img):
    em = DeepFace.analyze(img, actions=['age', 'gender', 'race', 'emotion'])
    os.remove(img)
    return em
   

video = cv2.VideoCapture(0)

while True:
    ret, img = video.read()
    if not ret:
        print("Camera error")
        break
    faces = face_tr.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        xaxis = int(x + w / 2)
        yaxis = int(y + h / 2)
        img = cv2.ellipse(img, center=(xaxis, yaxis), axes=(int(w/2), int(h/2)), angle=0, startAngle=0, endAngle=360, color=(255, 200, 0), thickness=4)
        em = det_gen(img)
        val = em[0]['dominant_gender']
        font = cv2.FONT_ITALIC
        cv2.putText(img, val, (x, y - 10), font, 2, (255, 0, 0), 2)
    cv2.imshow("show", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video.release()
cv2.destroyAllWindows()
