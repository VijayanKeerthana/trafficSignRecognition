# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 10:05:37 2021

@author: Devika Ajith
"""
from tensorflow.keras.models import load_model
import skimage
from skimage import transform, exposure, io
from imutils import paths
import numpy as np
import argparse
import imutils
import random
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import imageio 
import math
det_img_cnt = 0
image_path = "test_images_yolo/001_image1.png"
model_name = "trafficsignnet2.model"


def plot_img(image, label,size=[6.4,4.8]):
    plt.figure(figsize=size)
   # plt.axis(False)
    plt.title(label)
    plt.imshow(image)
    return

def predict_label(model, roi):
        
    # load the label names
    labelNames = open("signnames.csv").read().strip().split("\n")[1:]
    labelNames = [l.split(",")[1] for l in labelNames]
    #labelNames = ["prohibitory","danger","mandatory","other"]
    #predict
    image = transform.resize(roi, (32, 32))
    image = exposure.equalize_adapthist(image, clip_limit=0.1)
    
    # preprocess the image by scaling it to the range [0, 1]
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    
    # make predictions
    preds = model.predict(image)
    j = preds.argmax(axis=1)[0]
    print(j)
    label = labelNames[j]
    
    return label

def detected_image(image_path, bbs, labels):
    image = imageio.imread(image_path)
    for box,label in zip(bbs,labels):
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box[1] + box[3])), [172 , 10, 127], 2)
        cv2.putText(image, label, (int(box[0] + box[2]), int(box[1] + box[3])), cv2.FONT_HERSHEY_SIMPLEX,0.45, (0, 0, 255), 2)
        
    return image

def crop_roi(image,box):
    h, w = image.shape[:2]
    x_center, y_center = (box[0] * w), (box[1] * h)
    box_width, box_height = (box[2] * w), (box[3] * h)
    x_min, y_min = (x_center - box_width/2), (y_center - box_height/2)
    roi = image[int(y_min):int(y_min+box_height), int(x_min):int(x_min+box_width)]
    return roi, [x_min, y_min,box_width, box_height]

def yolo_detect_sign(captured_frame_path, count):
    
    output_path = "classified_images_cnn/classified%d.png" %count
    model = load_model(model_name)
    
    #yolo setup
    net = cv2.dnn.readNet("yolov4-tiny_training_1000.weights", "yolov4-tiny_training.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    confidence_threshold = 0.5
    
    #forward pass yolo
    image = cv2.imread(captured_frame_path)
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    
    outs = net.forward(output_layers)
    #reading again since classfier performing better on skimage
    image = skimage.io.imread(image_path)
    
    for out in outs:
        #     print(out.shape)
        for detection in out:
        #         print(len(detection))
            confidence = np.max(detection[5:])
            print('confidence', confidence)
            if confidence > confidence_threshold:
            #             print(confidence)
                roi, box = crop_roi(image,detection)
                confidences.append(float(confidence))
                rois.append(roi)
                boxes.append(box)
    
    #adjust overlaps
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    for i,roi in enumerate(rois):
        if i in indexes:
            label = predict_label(model, roi)
            plot_img(roi, label,[3,3])
            labels.append(label)
            bbs.append(boxes[i])
    
    # print(type(indexes[0]))
    # label_np=np.array(labels)
    # print(labels)
    # print(indexes)
    # print(len(bbs))
    # print(label_np[indexes])
    
    image = detected_image(image_path, bbs, labels)    
    plot_img(image, "final", [16,12])
    imageio.imsave(output_path,image)
""" 
    Displays detected and classified image from a video frame
"""
if __name__ == '__main__':
    class_ids = []
    confidences = []
    boxes = []
    rois = []
    labels = []
    bbs=[]
    count = 0
    captured_frame_path = ''
    cap = cv2.VideoCapture('video_data/001.mp4')
    frame_rate = cap.get(5)
    
    while (cap.isOpened): 
        frame_id = cap.get(1)
        ret, frame = cap.read()
        if (ret!= True): break
#        if (frame_id % math.floor(frame_rate) == 0):
#            count +=1
#            cv2.imshow('frame', frame)
            #vid_capture.set(cv2.CAP_PROP_POS_MSEC, (count*1000))
        cv2.imwrite("captured_frames/frame%d.jpg" %count, frame)
        captured_frame_path = 'captured_frames/frame'+str(count)+'.jpg'
        yolo_detect_sign(captured_frame_path, count)
        cv2.waitKey(6000)
#        if cv2.waitKey(50): # To Do: else endless loop need to fix
#            break
        
    vid_capture.release()
    cv2.destroyAllWindows()
