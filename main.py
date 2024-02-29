import cv2
import math
import numpy as np
import os
import torch
import os
import cvzone
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" #用來設置是否允許複製的 OpenMP 函數庫。 加入這行避免 import sort出現: OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.

from sort import *

from ultralytics import YOLO
# from utils.sort import Sort


def checkDevice():
    # Test cuda availability
    try:
        torch.cuda.is_available()
    except:
        device = 'cpu'
    else:
        device = 'cuda:0'
    finally:
        print('Running on %s' % device)
        return device
    
def checkVideo(videoPath):
    if not os.path.exists(videoPath):
        print('Video not found')
        exit()
    else:
        video = cv2.VideoCapture(videoPath)
        return video



def draw_boxes(img, className, pred, color=(255, 0, 255)):
    global detections
    for result in pred:
        for box in result.boxes:
            # Get the coordinates of the box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # Convert to int
            w, h = x2 - x1, y2 - y1
            # Get the confidence score
            conf = math.ceil(box.conf[0] * 100) / 100
            # Get the predicted class label
            cls = className[int(box.cls[0])]

            if (cls == 'car' or cls == 'truck' or cls == 'bus') and conf > 0.3:
                # Draw the box
                #cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                # Draw the label
                #cv2.putText(img, '%s %.2f' % (cls, conf), (x1, y1 - 10),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections , currentArray))
    return img
    



def main(videoPath, modelName):
    global detections , totalcount
    device = checkDevice()  # Check device for running the model
    model = YOLO(modelName).to(device)  # Load model
    video = checkVideo(videoPath)  # Load video
    classes = ["person", "bicycle", "car", "motorbike", "airplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed",
              "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]  # class list for COCO dataset
    
    # Loop
    while True:
        success, frame = video.read()  # Read frame
        if not success:
            break

        imgRegion = cv2.bitwise_and(frame , mask_resized)

        #add car graph
        imggraphics = cv2.imread(car_graph_path , cv2.IMREAD_UNCHANGED)
        frame = cvzone.overlayPNG(frame , imggraphics , (0, 0))

        # Detect
        results = model(imgRegion,verbose=False) # result list of detections

        #detections
        detections = np.empty((0 , 5)) #這個陣列的第一維度為0，表示陣列一開始是空的，而第二維度為5，表示每個元素包含5個值(x1,y1,x2,y2,score)

        # Draw
        frame = draw_boxes(frame, classes, results)

        #track
        resultTracker = tracker.update(detections)

        cv2.line(frame,(limits[0],limits[1]) , (limits[2],limits[3]) , (0 , 0 , 255) , 5)


        for result in resultTracker:
            x1 , y1 , x2 , y2 , id = result
            x1 , y1 , x2 , y2 = int(x1) , int(y1) , int(x2) , int(y2)
            #print(result)
            w , h =x2 - x1 , y2 - y1
            # Draw the box
            cvzone.cornerRect(frame , (x1 , y1 , w, h) , l = 9 ,rt =2 , colorR=(255 , 0, 0))
            # Draw the label
            cvzone.putTextRect(frame , f'{int(id)}' , (max(0,x1),max(35,y1)),scale = 2 , thickness = 3 , offset = 10)

            
            cx , cy = x1 + w//2 , y1 + h//2
            cv2.circle(frame , (cx , cy) , 5 , (255 , 0 , 255) , cv2.FILLED)

            if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15 :
                if totalcount.count(id) == 0: #數totalcount的陣列中，某id出現的次數。若為0次則加入totalcount的陣列
                    totalcount.append(id)
                    cv2.line(frame,(limits[0],limits[1]) , (limits[2],limits[3]) , (0 , 255 , 0) , 5)
                
        #cvzone.putTextRect(frame , f' Count :{len(totalcount)}' , (50,50) )
                    
        cv2.putText(frame , str(len(totalcount)) , (255 , 100) , cv2.FONT_HERSHEY_PLAIN , 5 , (50 , 50 ,255), 8 )

        # Show
        cv2.imshow('frame', frame)
        #cv2.imshow('imgRegion', imgRegion)
        cv2.waitKey(1)
        # Break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # close all windows
    cv2.destroyAllWindows()
    os.system('clear')



if __name__ == '__main__': 
    videoPath = './共同交接week5(yolov8)/cars.mp4'
    modelName = './共同交接week5(yolov8)/yolov8n.pt'
    mask_path =  "./共同交接week5(yolov8)/ObjectDetect/mask.png"
    car_graph_path = "./共同交接week5(yolov8)/ObjectDetect/graphics.png"
    mask = cv2.imread(mask_path)
    if mask is None:
        print("fail to read mask")
        exit()
    mask_resized = cv2.resize(mask, (1280, 720))

    #tracking
    tracker = Sort(max_age = 20 , min_hits = 3 , iou_threshold = 0.3)

    #limit
    limits = [400,297,673,297]

    #totalcount的陣列
    totalcount = []
    
    main(videoPath, modelName)