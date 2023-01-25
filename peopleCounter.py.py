import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.pyplot as plt
image = plt.imread("people1.jpg")
classes = None
#Consists of the names of the objects that this algorithm can detect
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

#x and y axis or length or breadth
Width = image.shape[1]
Height = image.shape[0]

# read pre-trained model and config file
#deep neural network

#The trained model that detects the objects.
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# create input blob  binary large object
#preprocessing of the image is done
#0.00392 is scale factor
#416*416 is the size of the image
#no mean subtraction se we have taken (0,0,0)
#true means SwapRB
net.setInput(cv2.dnn.blobFromImage(image, 0.00392, (416,416), (0,0,0), True, crop=False))

# run inference through the network
# and gather predictions from output layers
i = []
#getLayerNames() function
#and getUnconnectedOutLayers() function to get the output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
outs = net.forward(output_layers)

#index assigning [0,1,2]
class_ids = []
confidences = []
#x,y coordinates of center,w,h
boxes = []

#create bounding box 
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.9:
            #detection[0] is x cordinate
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

#non maximum supression for removing multiple boxes
#first 0.1 is the confidential threshold
#second 0.1 is nms threshold
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.9, 0.1)

#check if is people detection 
#drawing boxes
for i in indices:
    box = boxes[i]
    if class_ids[i]==0:
        label = str(classes[class_id]) 
        cv2.rectangle(image, (round(box[0]),round(box[1])), (round(box[0]+box[2]),round(box[1]+box[3])), (0, 0, 0), 2)
        #cv2.putText(image, label, (round(box[0])-10,round(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
cv2.putText(image, "Count =" + str(len(indices)), (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
plt.imshow(image)
print(len(indices))
cv2.imshow('Pedestrians', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

classes = None
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# read pre-trained model and config file
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# create input blob 
# set input blob for the network
cap = cv2.VideoCapture("test.mp4")

# Loop once video is successfully loaded
while cap.isOpened():
    
    # Read first frame
    ret, img = cap.read()
    scale_percent = 60 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
  
    # resize image
    image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    Width = image.shape[1]
    Height = image.shape[0]

    net.setInput(cv2.dnn.blobFromImage(image, 0.00392, (416,416), (0,0,0), True, crop=False))

    # run inference through the network
    # and gather predictions from output layers
    i = []
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(output_layers)
    

    class_ids = []
    confidences = []
    boxes = []

#create bounding box 
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.9:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])


    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.9, 0.1)

#check if is people detection
    for i in indices:
        box = boxes[i]
        if class_ids[i]==0:
            label = str(classes[class_id]) 
            cv2.rectangle(image, (round(box[0]),round(box[1])), (round(box[0]+box[2]),round(box[1]+box[3])), (0, 0, 0), 2)
            #cv2.putText(image, label, (round(box[0])-10,round(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    
    cv2.putText(image, "Count =" + str(len(indices)), (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.imshow('Pedestrians', image)
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
#print(len(indices))
cap.release()
cv2.destroyAllWindows()
