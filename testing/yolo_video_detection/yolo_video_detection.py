# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 10:05:37 2021

@author: devji
"""

import cv2
import numpy as np

classes = ['prohibitory', 'danger', 'mandatory', 'other']

#Detection model
def yolo_detect_sign(img_path):
   

# Images path
#images_path = glob.glob(r"../images/input_images/*.jpg")
    print(img_path)


    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Insert here the path of your images
    #random.shuffle(images_path)
# loop through all the images
#for img_path in images_path:
    # Loading image
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.1:
                # Object detected
                print(class_id)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    scale = 1
    fontScale = min(416,416)/(25/scale)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 2)


    cv2.imshow("Image", img)
    key = cv2.waitKey(0)

    cv2.destroyAllWindows()

#Classification model
def traffic_sign_net_predict():
    pass
if __name__ == '__main__':

    #video capture
    vid_capture = cv2.VideoCapture('video_data/2.mp4')
    
    success, image = vid_capture.read()
    count = 0
    net = cv2.dnn.readNet("yolov4-tiny_training_1000.weights", "yolov4-tiny_training.cfg")
    classes = []
    with open("signs.names.txt", "r") as f:
        classes = [line.strip() for line in f.readlines()]
        
     #get last layers names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    check_time = True
    confidence_threshold = 0.5
    font = cv2.FONT_HERSHEY_SIMPLEX
    detection_confidence = 0.5
    
    while success:
        print('success')
        vid_capture.set(cv2.CAP_PROP_POS_MSEC, (count*1000))
        cv2.imwrite("captured_frames/frame%d.jpg" %count, image)
        path = 'captured_frames/frame'+str(count)+'.jpg'
        yolo_detect_sign(path)
        
        if count == 30: #else endless loop need to fix
            break
        count +=1
    vid_capture.release()
    cv2.destroyAllWindows()
