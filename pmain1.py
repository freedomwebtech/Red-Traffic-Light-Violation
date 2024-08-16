import cv2
from ultralytics import YOLO
import pandas as pd
import cvzone
import numpy as np
from test1 import process_frame
import os
from tracker import*
from datetime import datetime

model = YOLO("yolov10s.pt")  

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('tr.mp4')
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
tracker=Tracker()
count = 0
area = [(324, 313), (283, 374), (854, 392), (864, 322)]

# Create directory for today's date
today_date = datetime.now().strftime('%Y-%m-%d')
output_dir = os.path.join('saved_images', today_date)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
list1=[]
while True:
    ret, frame = cap.read()
    count += 1
    if count % 2 != 0:
        continue
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 600))
    processed_frame, detected_label = process_frame(frame)
    
    results = model(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    list=[]
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        
        d = int(row[5])
        c = class_list[d]
        list.append([x1,y1,x2,y2])
    bbox_idx=tracker.update(list)
    for bbox in bbox_idx:
        x3,y3,x4,y4,id=bbox
    
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
#        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
        result = cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False)
        if result>=0:
           if 'car' in c and detected_label == "RED":
                cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
                
                # Save the image with red label
                timestamp = datetime.now().strftime('%H-%M-%S')
                image_filename = f"{timestamp}.jpg"
                output_path = os.path.join(output_dir, image_filename)
                if list1.count(id)==0:
                   list1.append(id)
                   cv2.imwrite(output_path, frame)
           else:     
                cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                
    

           

    cv2.polylines(frame, [np.array(area, np.int32)], True, (0, 255, 0), 2)
    cv2.imshow("RGB", frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
