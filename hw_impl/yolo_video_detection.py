# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 10:05:37 2021

@author: Devika Ajith
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model
#from skimage import io
import imageio
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import os
classes = ['prohibitory', 'danger', 'mandatory', 'other']
detected_img = 0

"""
    Captures traffic sign from a video frame, crops it and writes it to a folder
"""
def yolo_detect_sign(img_path):

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    
    # Object detection
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.1:
                center_x = int(detection[0] * width)
                
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    scale = 1
    fontScale = min(416,416)/(25/scale)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            
        
            cv2.putText(img, label, (x, y + 30), font, 3, color, 1)
        roi = img[y:y+h, x:x+w]
        tbd_image_path = "test_images_cnn/00004.png"
        output_path = "classified_images_cnn/classified%d.jpg" %i
        cv2.imwrite("detected_images_yolo/detected%d.jpg" %i, roi)
        model_name = 'trafficsignnet2.model'
        prediction(model_name, tbd_image_path, output_path)
    cv2.imshow("Image", img)
    
    key = cv2.waitKey(0)

    cv2.destroyAllWindows()
    
    
""" 
    Classifies the image and sounds a buzzer
"""
def crop_box(image,line):
    
    #split the line
    box = [float(i) for i in line.split()]
    h, w = image.shape[:2]
    
    #need to multiply with width/height since bounding boxes are in percentage
    x_center, y_center = (box[1] * w), (box[2] * h)
    box_width, box_height = (box[3] * w), (box[4] * h)
    x_min, y_min = (x_center - box_width/2), (y_center - box_height/2)
    
    #cropping region of interest
    roi = image[int(y_min):int(y_min+box_height), int(x_min):int(x_min+box_width)]
    
    return roi, [x_min, y_min,box_width, box_height]
#function to predict label for each input box

def predict_label(model, roi):
        
    # load the label names
    labelNames = open("signnames.csv").read().strip().split("\n")[1:]
    labelNames = [l.split(",")[1] for l in labelNames]
    
    #predict
    image = cv2.resize(roi, (32, 32))
    #image = exposure.equalize_adapthist(image, clip_limit=0.1)
    
    # preprocess the image by scaling it to the range [0, 1]
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    
    # make predictions
    preds = model.predict(image)
    j = preds.argmax(axis=1)[0]
    label = labelNames[j]
    
    return label
#function to draw boxes and labels on the first input image

def detected_image(image_path, bbs, labels):
    image = imageio.imread(image_path)
    for box,label in zip(bbs,labels):
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box[1] + box[3])), [172 , 10, 127], 2)
        cv2.putText(image, label, (int(box[0] + box[2]), int(box[1] + box[3])), cv2.FONT_HERSHEY_SIMPLEX,0.45, (0, 0, 255), 2)
        
    return image
def plot_img(image, label,size=[6.4,4.8]):
    plt.figure(figsize=size)
    #plt.axis(False)
    plt.title(label)
    plt.imshow(image)
    return
def prediction(model_name, tbd_image_path,detected_output_path):
    print("[INFO] loading model...")
    model = load_model(model_name)
    
    print("[INFO] predicting...")
    input_image = imageio.imread(tbd_image_path)

    plot_img(input_image, tbd_image_path.split("\\")[-1],[12,9])

    #since YOLO dataset has similar naming structure
    txt_path = tbd_image_path[:-3]+"txt"
    boxes=[]
    labels = []
    with open (txt_path, "r+") as f:
        for line in f:
            roi, box = crop_box(input_image,line)
            boxes.append(box)

            label = predict_label(model, roi)
            labels.append(label)

            plot_img(roi, label,[3,3])
        
    image = detected_image(tbd_image_path, boxes, labels)    
    plot_img(image, "final", [16,12])
    
    #save in an output folder
    imageio.imsave(detected_output_path,image)
   # detected_img = detected_img + 1
    return

""" 
    Displays detected and classified image from a video frame
"""
if __name__ == '__main__':
    vid_capture = cv2.VideoCapture('video_data/2.mp4')
    success, image = vid_capture.read()
    count = 0
    net = cv2.dnn.readNet("yolov4-tiny_training_1000.weights", "yolov4-tiny_training.cfg")
    
#    classes = []
#    with open("signs.names.txt", "r") as f:
#        classes = [line.strip() for line in f.readlines()]
#        
#     #get last layers names
#    layer_names = net.getLayerNames()
#    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
#    colors = np.random.uniform(0, 255, size=(len(classes), 3))
#    check_time = True
#    confidence_threshold = 0.5
#    font = cv2.FONT_HERSHEY_SIMPLEX
#    detection_confidence = 0.5
#    
    while success:
        vid_capture.set(cv2.CAP_PROP_POS_MSEC, (count*1000))
        cv2.imwrite("captured_frames/frame%d.jpg" %count, image)
        path = 'captured_frames/frame'+str(count)+'.jpg'
        yolo_detect_sign(path)
        if count == 30: # To Do: else endless loop need to fix
            break
        count +=1
    vid_capture.release()
    cv2.destroyAllWindows()
