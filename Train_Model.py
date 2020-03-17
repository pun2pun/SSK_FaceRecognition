

import cv2
import os
import numpy as np
from PIL import Image


def trainModel():
        path = 'dataset'
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        detector = cv2.CascadeClassifier("Material/haarcascade_frontalface_default.xml")
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     

        faceSamples=[]
        ids = []

        for imagePath in imagePaths:

            PIL_img = Image.open(imagePath).convert('L') 
            img_numpy = np.array(PIL_img,'uint8')

            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)
            #print('Id : ',id)
            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)
        recognizer.train(faceSamples, np.array(ids))
        recognizer.write('trainModel/Model.yml')

        print('Update model already !')
