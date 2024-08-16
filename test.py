import cv2
import numpy as np


cap = cv2.VideoCapture('tr.mp4')

def process_frame(frame):
    # Define the color ranges
    lower_range = np.array([58, 97, 222])  # Green color range
    upper_range = np.array([179, 255, 255])
    lower_range1 = np.array([0, 43, 184])  # Red color range
    upper_range1 = np.array([56,132, 255])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks for both color ranges
    mask = cv2.inRange(hsv, lower_range, upper_range)
    mask1 = cv2.inRange(hsv, lower_range1, upper_range1)

    # Combine the two masks
    combined_mask = cv2.bitwise_or(mask, mask1)

    # Threshold the combined mask
    _, final_mask = cv2.threshold(combined_mask, 254, 255, cv2.THRESH_BINARY)
    detected_label = None

    # Find contours
    cnts, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in cnts:
        if cv2.contourArea(c) > 50:
            x, y, w, h = cv2.boundingRect(c)
            
            # Calculate the center point of the rectangle
            cx = x + w // 2
            cy = y + h // 2
    

            # Determine the color of the contour
            if cv2.countNonZero(mask[y:y+h, x:x+w]) > 0:  # Green color range
               color = (0, 255, 0)  # Green color for the rectangle
               text_color = (0, 255, 0)  # Green text
               label = "GREEN"
            elif cv2.countNonZero(mask1[y:y+h, x:x+w]) > 0:  # Red color range
                color = (0, 0, 255)  # Red color for the rectangle
                text_color = (0, 0, 255)  # Red text
                label = "RED"
            else:
                continue
            
            detected_label = label 
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
            # Draw the center point
            cv2.circle(frame, (cx, cy), 1, (255, 0, 0), -1)  # Draw a small blue circle at the center
            
            # Display text
            cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    
    return frame,detected_label

count = 0




while True:
    ret, frame = cap.read()
    count += 1
    if count % 3 != 0:
        continue
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 600))
    processed_frame,detected_label = process_frame(frame)
    


    cv2.imshow("RGB", frame)
    if cv2.waitKey(0) & 0xFF == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()