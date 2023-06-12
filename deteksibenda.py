import cv2
import torch 
import numpy as np 

#kelompok 6 kecerdasan buatan d081
#Farrel Tiuraka Vierino 21081010222
#M. Utbah Husnuth Thoriq 21081010131
#Albert Vincentius 21081010212

model = torch.hub.load('ultralytics/yolov5', 'custom', 'last.onnx')

know_distance = 40
know_width = 20

GREEN = (0, 255, 0)
RED = (0,0,255)
WHITE = (255,255,255)
BLACK = (0,0,0)

font = cv2.FONT_HERSHEY_COMPLEX

def Focal_Length_Finder(measured_distance, real_width, width_in_rf_image):
    return (width_in_rf_image * measured_distance) / real_width

def Distance_Finder(Focal_Length, real_face_width, face_width_in_frame):
    return (real_face_width * Focal_Length) / face_width_in_frame
    
ref_image = cv2.imread("image\bl.jpg")
ref_image = cv2.resize(ref_image, (350, 350))

result = model(ref_image)
result.print()
for result in result.xyxy[0]:
  if result[5] == 0:
    x1, y1, x2, y2, conf, cls_conf= result
    x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)

    Focal_Length_Found = Focal_Length_Finder(know_distance, know_width, w)
    print(Focal_Length_Found)

    cv2.putText(ref_image, "Ball", (x, y -10), cv2.FONT_HERSHEY_PLAIN, 2, (200,0,50), 2)
    
    cv2.rectangle(ref_image, (x,y), (x+w, y+h), (0,255,0), 2)
    
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    frame = cv2.resize(frame, (350, 350))

    result = model(frame)
    for result in result.xyxy[0]:
        if result[5] == 0:
            x1, y1, x2, y2, conf, cls_conf= result
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            className = "Ball"

            distance = Distance_Finder(Focal_Length_Found, know_width, w)
            titikX = int(x + w/2)
            titiY = int(y + h/2)
            
            JarakX = int(x + w/2) - 175
            print(JarakX)
            
            cv2.putText(frame, "Ball", (x, y -10), cv2.FONT_HERSHEY_PLAIN,  2, (200,0,50), 2)
            cv2.putText(frame, "+", (titikX, titiY), font, 0.6, BLACK, 2)   
            cv2.putText(frame, f"Distance: {round(distance, 2)} CM", (30,33), font, 0.6, GREEN, 2)

            cv2.line(frame, (titikX,175), (175, 175), WHITE, 2)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

    cv2.putText(frame, "+", (175, 175), font, 0.6, RED, 2)       
    cv2.imshow('webcam', frame)
    

    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()
