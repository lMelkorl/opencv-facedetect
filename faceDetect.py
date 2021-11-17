import cv2
import numpy as np
import os

camera = cv2.VideoCapture(0)

while (True):
    ret , cam = camera.read()
    grayk = cv2.cvtColor(cam , cv2.COLOR_BGR2GRAY)

    faceClass =  cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    result = faceClass.detectMultiScale(grayk , 1.3 , 5)
    for (x,y,gen,yuk) in result:
        cv2.rectangle(cam , (x,y) , (x+gen , y + yuk) , (0,0,255) , 2)

    cv2.imshow("kare" , cam)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()