import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pygame

# =============================
# LOAD MODEL
# =============================
model = load_model("drowsiness_model.h5")

# =============================
# LOAD CASCADE FILES
# =============================
face_cascade = cv2.CascadeClassifier("haar cascade files/haarcascade_frontalface_alt.xml")
left_eye_cascade = cv2.CascadeClassifier("haar cascade files/haarcascade_lefteye_2splits.xml")
right_eye_cascade = cv2.CascadeClassifier("haar cascade files/haarcascade_righteye_2splits.xml")

# =============================
# LOAD ALARM SOUND
# =============================
pygame.mixer.init()
pygame.mixer.music.load("alarm.wav")

# =============================
# START CAMERA
# =============================
cap = cv2.VideoCapture(0)

score = 0
status = ""

while True:

    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    left_eye = left_eye_cascade.detectMultiScale(gray)
    right_eye = right_eye_cascade.detectMultiScale(gray)

    # =============================
    # FACE RECTANGLE
    # =============================
    for (x, y, w, h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(100,100,100),1)

    # =============================
    # RIGHT EYE DETECTION
    # =============================
    for (x,y,w,h) in right_eye:

        r_eye = frame[y:y+h, x:x+w]
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))
        r_eye = r_eye / 255
        r_eye = r_eye.reshape(1,24,24,1)

        prediction = model.predict(r_eye, verbose=0)

        if prediction[0][0] > 0.5:
            status = "Closed"
        else:
            status = "Open"

        break

    # =============================
    # LEFT EYE DETECTION
    # =============================
    for (x,y,w,h) in left_eye:

        l_eye = frame[y:y+h, x:x+w]
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye = l_eye / 255
        l_eye = l_eye.reshape(1,24,24,1)

        prediction = model.predict(l_eye, verbose=0)

        if prediction[0][0] > 0.5:
            status = "Closed"
        else:
            status = "Open"

        break

    # =============================
    # DROWSINESS LOGIC
    # =============================
    if status == "Closed":
        score += 1
        cv2.putText(frame,"Eyes Closed",(10,height-20),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

    else:
        score -= 1
        cv2.putText(frame,"Eyes Open",(10,height-20),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    if score < 0:
        score = 0

    cv2.putText(frame,'Score:'+str(score),(100,20),
                cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1)

    # =============================
    # ALARM TRIGGER
    # =============================
    if score > 15:

        cv2.putText(frame,"DROWSINESS ALERT!",(100,200),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.play()

    # =============================
    # DISPLAY WINDOW
    # =============================
    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
