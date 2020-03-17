import cv2

def begin_scan():
        
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        faceCascade = cv2.CascadeClassifier("Material/haarcascade_frontalface_default.xml")
        out = cv2.VideoWriter('output.mp4', -1, 20.0, (640,360))
        try:
            recognizer.read('trainModel/model.yml')
        except:
            print('Please check you file Model or Casecade')


        font = cv2.FONT_HERSHEY_SIMPLEX
        
        cam = cv2.VideoCapture('avenger_vdo.mp4')
        cam.set(3, 640) # set video widht
        cam.set(4, 480) # set video height

        # Define min window size to be recognized as a face
        minW = 0.1*cam.get(3)
        minH = 0.1*cam.get(4)
     

        while True:
            ret, img =cam.read()
            
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale(gray,
                scaleFactor = 1.2,
                minNeighbors = 5,
                minSize = (int(minW), int(minH)),
               )
            
            for(x,y,w,h) in faces:
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

                id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
                print(confidence)
                if (confidence < 90 ):
                    
                    confidence = "  {0}%".format(round(100 - confidence))  
                   
                    if(id == 10):
                        name = 'Back Panter'

                    if(id == 20):
                        name = 'Buckky'

                    if(id == 30):
                        name = 'Captain'

                    cv2.putText(img, str(name), (x+5,y-5), font, 1, (0,255,0), 2)
                    cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)                                                    
                    
                    #cv2.imwrite('image.jpg',img) 
                        
                else:
                    name = "unknown"
                    confidence = "  {0}%".format(round( 100 - confidence ))
                
                
                
                
            window_name = 'camera'
            cv2.imshow(window_name,img) 
            print(img.shape)
            out.write(img)

            k = cv2.waitKey(10) & 0xff 
            if k == 27:        # Press 'ESC' for exiting video
                break

       
        
        cam.release()
        cv2.destroyAllWindows()





begin_scan()