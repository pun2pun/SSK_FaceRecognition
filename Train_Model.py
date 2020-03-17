
import cv2
import os
import numpy as np
from PIL import Image

def trainModel():
        path = 'dataset'                                                                           # โฟรเดอร์ที่ใช้เก็บ
        recognizer = cv2.face.LBPHFaceRecognizer_create()                                          # เรียกใชฟังก์ชั่นการจดจำใบหน้า                            
        detector = cv2.CascadeClassifier("Material/haarcascade_frontalface_default.xml")        #เรียนกใช้ไฟล์ Cascade ของใบหน้า
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]                            # อ่านรายชื่อไฟล์ใน path

        faceSamples=[]                                                                             # ประกาศ  faceSamples ให้เป็นตัวแปรแบบ Array
        ids = []

        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L')                                        # แปลงรูปภาพให้เป็นตัวเลข                
            img_numpy = np.array(PIL_img,'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])                               # ดึงค่า id จากชื่อไฟล์รูป
            faces = detector.detectMultiScale(img_numpy)
            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])                                      # เก็บค่าตำแหน่งบหน้าไปใน faceSamples
                ids.append(id)                                                                  # เก็บค่า id บหน้าไปใน ids
        recognizer.train(faceSamples, np.array(ids))                                            # Train Model โดยใช้   faceSamples ร่วมกับ ids
        recognizer.write('trainModel/Model.yml')                                                # บันทึกไฟล์ Model ไปที่โฟรเดอร์ trainModel และใช้ไฟล์ชื่อ Model.yml 

        print('Update model already !')

