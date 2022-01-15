# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 10:05:37 2021

@author: 
"""
import numpy as np
import cv2
import time
#from grabscreen import grab_screen
from tensorflow.keras.models import load_model

classes = ['prohibitory', 'danger', 'mandatory', 'other']
detected_img = 0

""" 
    Displays detected and classified image from a video frame
"""
if __name__ == '__main__':
    '''Input size expected by the classification_model'''
    HEIGHT = 32
    WIDTH = 32
    ''' Load Yolo'''
    net = cv2.dnn.readNet("yolov4-tiny_training_1000.weights", "yolov4-tiny_training.cfg")
    
    classes = []
    with open("signsnames.txt", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    check_time = True
    confidence_threshold = 0.5
    font = cv2.FONT_HERSHEY_SIMPLEX
    start_time = time.time()
    frame_count = 0
    
    detection_confidence = 0.5
    cap = cv2.VideoCapture(0)
    
    '''Load Classification Model'''
    classification_model = load_model('traffic.h5') #load mask detection model
    classes_classification = []
    with open("classifiersignnames.txt", "r") as f:
        classes_classification = [line.strip() for line in f.readlines()]
    
    tfps = 30
    cap = cv2.VideoCapture('video_data/4.mp4')
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    hop = round(fps/tfps)
    curr_frame = 0
    while(True):
        ret, frame = cap.read()
        if not ret:
#            print('nooo')
            break
#        if curr_frame % hop == 0:
#        print('capturing frames')
        path = 'captured_frames/frame'+str(curr_frame)+'.jpg'
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        #get image shape
        frame_count += 1
        height, width, channels = img.shape
        window_width = width
        
        #detecting objects (YOLO)
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        
        # Show bounding box on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > confidence_threshold:
                    # Object detected
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
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]]) + "=" + str(round(confidences[i]*100, 2)) + "%"
                cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 2)
                '''crop the detected signs -> input to the classification model'''
                crop_img = img[y:y+h, x:x+w]
                save_img = crop_img
                if len(crop_img) >0:
                    
#                    print('image should show up herer')
                    crop_img = cv2.resize(crop_img, (WIDTH, HEIGHT))
                    crop_img =  crop_img.reshape(-1, WIDTH,HEIGHT,3)
                    prediction = np.argmax(classification_model.predict(crop_img))
                    label = str(classes_classification[prediction])
                    print(label)
                    cv2.putText(img, label, (x, y), font, 0.5, (255,0,0), 2)
                    cv2.imwrite("classified_images_cnn/" + label + ".jpg", save_img)
        elapsed_time = time.time() - start_time
        fps = frame_count/elapsed_time
        print ("fps: ", str(round(fps, 2)))
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord ('q'):
            break
    cv2.destroyAllWindows()
        
        
