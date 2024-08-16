import cv2
from ultralytics import YOLO
import pandas as pd
import cvzone
import numpy as np
from test1 import process_frame
model = YOLO("yolov10s.pt")  




    

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)


cap=cv2.VideoCapture('tr.mp4')
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")


count=0
area=[(324,313),(283,374),(854,392),(864,322)]
while True:
    ret,frame = cap.read()
    count += 1
    if count % 3 != 0:
        continue
    if not ret:
       break
    
    frame = cv2.resize(frame, (1020,600))
    processed_frame,detected_label = process_frame(frame)
    
    results = model(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        
        d = int(row[5])
        c = class_list[d]
        cx=int(x1+x2)//2
        cy=int(y1+y2)//2
             
        result=cv2.pointPolygonTest(np.array(area,np.int32),((cx,cy)),False)
        if result>=0:
           if 'car' in c  and detected_label=="GREEN":
              cvzone.putTextRect(frame,f'{c}',(x1,y1),1,1)
              cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            
        result1=cv2.pointPolygonTest(np.array(area,np.int32),((cx,cy)),False)
        if result1>=0:
           if 'car' in c  and detected_label=="RED":
                cvzone.putTextRect(frame,f'{c}',(x1,y1),1,1)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)  
            
     


         

                      
                 
                 
    
    cv2.polylines(frame,[np.array(area,np.int32)],True,(0,255,0),2)
    cv2.imshow("RGB", frame)
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()


