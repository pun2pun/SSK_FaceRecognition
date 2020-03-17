import cv2


def tell_Me_is_who(img,gray,faces,recognizer):
    for(x,y,w,h) in faces:                                                  
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)                        # สร้างกรอบสี่เหลี่ยมที่ตำแหน่ง x,y และมีขนาดเท่ากับ w,h สีเขียว ความหนาเส้น 2 pixel
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])                    # เรียกใช้ predict ละจะได้ค่า id และ confidence กลับมา
        
        if (confidence < 90 ):                                                    # ตั้งเงื่อนไขตามความเหมาะสม
            confidence = "  {0}%".format(round(100 - confidence))                 # ปรับค่า confidence ให้เป็น %
            if(id == 10):
                name = 'Back Panter'
            if(id == 20):
                name = 'Buckky'
            if(id == 30):
                name = 'Captain'

            font = cv2.FONT_HERSHEY_SIMPLEX                                          # เลือก font ให้ตัวหนังสือ 
            cv2.putText(img, str(name), (x+5,y-5), font, 1, (0,255,0), 2)            # เพิ่มหนังสือ โดยให้เป็นชื่อ
            cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)   # เพิ่มหนังสือ โดยให้เป็นความแม่นยำ                                                                                          
        else:
            name = "unknown"
            confidence = "  {0}%".format(round( 100 - confidence ))
    return img


##-----------------------------------------------------------------------------------------------------------


def begin_scan():
        
        faceCascade = cv2.CascadeClassifier("Material/haarcascade_frontalface_default.xml")   #เรียนกใช้ไฟล์ Cascade ของใบหน้า
        
        recognizer = cv2.face.LBPHFaceRecognizer_create()                           # เรียกใชฟังก์ชั่นการจดจำใบหน้า
        recognizer.read('trainModel/Model.yml')                                     # โหลดค่ารูปแบบ Model ของใบหน้า
       
        cam = cv2.VideoCapture('avenger_vdo.mp4')                        # กำหนดแหล่งที่มาของ Vedio
        cam.set(3, 640)                                                  # กำหนดขนาดความยาว = 640
        cam.set(4, 480)                                                  # กำหนดขนาดความกว้าง = 480

        # Define min window size to be recognized as a face
        minW = 0.1*cam.get(3)                                        
        minH = 0.1*cam.get(4)

        while True:
            ret, img =cam.read()                                          # อ่านค่า frame จาก Vedio
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)                   # เปลียนรูปสี ให้เป็นแบบขาวดำ(Gray scale)

            faces = faceCascade.detectMultiScale(gray,                     
                scaleFactor = 1.2,
                minNeighbors = 5,
                minSize = (int(minW), int(minH)),
               )
                       
            img = tell_Me_is_who(img,gray,faces,recognizer)            # เรียกใช้ฟังก์ชั่น tell_Me_is_who ที่เราสร้างขึ้น
                           
            window_name = 'camera'
            cv2.imshow(window_name,img) 

            k = cv2.waitKey(10) & 0xff 
            if k == 27:                                                  # Press 'ESC' for exiting video
                break

        cam.release()
        cv2.destroyAllWindows()

begin_scan()






