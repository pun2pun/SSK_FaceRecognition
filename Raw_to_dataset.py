import cv2
import os
import numpy as np
from PIL import Image


def make_to_gray():
    folder = 'Raw_data_set'
    files =  os.listdir(folder)
    face_detector = cv2.CascadeClassifier('Material/haarcascade_frontalface_default.xml')
    
    for name in files:
        img = cv2.imread('Raw_data_set/' + name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            gray = gray[y:y+h,x:x+w]
            cv2.imwrite('dataset/' + name,gray)

    print('Complete !')

