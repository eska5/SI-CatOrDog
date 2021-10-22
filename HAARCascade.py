import cv2
import numpy as np
import time
# pobrane cascady
hiro_cascade = cv2.CascadeClassifier('hiro.xml')

img = cv2.imread('hiro_akwarium.bmp', cv2.IMREAD_COLOR)
img = cv2.resize(img, (1000, 1000))
#cap = cv2.VideoCapture(0)

#ret, img = cap.read()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = hiro_cascade.detectMultiScale(gray, 1.05, 7, minSize=(20, 20))

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
cv2.imshow('img', img)
k = cv2.waitKey(30) & 0xff
# if k == 27:
#	break
time.sleep(5)
# cap.release()
cv2.destroyAllWindows()
