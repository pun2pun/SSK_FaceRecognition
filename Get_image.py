
import cv2
import os
import numpy as np
from PIL import Image


def Start_get_data(ids):
        
        cam = cv2.VideoCapture(0)
        cam.set(3, 400)
        cam.set(4, 400) 
        face_detector = cv2.CascadeClassifier('Material/haarcascade_frontalface_default.xml')
        face_id =  ids
        image_number = 20
        count = 0
        window_name = 'Get new face'
        ofset_crop = 10
        #print(face_id)
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 2)
            for (x,y,w,h) in faces:
                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)   

                font = cv2.FONT_HERSHEY_SIMPLEX
                percen_process = " {0:0.2f}%".format(round( (count/image_number) * 100 ))
                cv2.putText(img, percen_process , (x+5,y-5), font, 1, (255,255,255), 2)  
                count += 1
                cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h+ofset_crop,x:x+w+ofset_crop])
            
            cv2.imshow(window_name, img)
            k = cv2.waitKey(100) & 0xff
            if k == 27:
                break
            elif count >= image_number: 
                 break
        cam.release()
        cv2.destroyAllWindows()     
        print('Get data complete')

Start_get_data('30')