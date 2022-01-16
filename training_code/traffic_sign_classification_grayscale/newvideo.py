# import the necessary packages
from tracemalloc import start
from tensorflow.keras.models import load_model
from skimage import transform
from skimage import exposure
from skimage import io
from imutils import paths
import numpy as np
import argparse
import imutils
import random
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

#function to predict label for each input box

def predict_label(model, roi, labelNames):
    try:
        image = cv2.resize(roi,(32, 32))
    except:
        return "0"
    image = image[np.newaxis,:,:,np.newaxis]
    image = np.array(image).astype("float32") / 255.0
    
    # make predictions
    preds = model.predict(image)
    if np.max(preds)>0.75:
        j = preds.argmax(axis=1)[0]
        label = labelNames[j]
    else:
        label = "others"
    
    return label
    
#function to draw boxes and labels on the first input image

def detected_image(image_path, bbs, labels):
    image = io.imread(image_path)
    for box,label in zip(bbs,labels):
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box[1] + box[3])), [172 , 10, 127], 2)
        cv2.putText(image, label, (int(box[0] + box[2]), int(box[1] + box[3])), cv2.FONT_HERSHEY_DUPLEX,0.75, (255, 255, 0), 2)
        
    return image
    
#function to crop box
def crop_roi(image,box):
    h, w = image.shape[:2]
    x_center, y_center = (box[0] * w), (box[1] * h)
    box_width, box_height = (box[2] * w), (box[3] * h)
    x_min, y_min = (x_center - box_width/2), (y_center - box_height/2)
    roi = image[int(y_min):int(y_min+box_height), int(x_min):int(x_min+box_width)]
    return roi, [x_min, y_min,box_width, box_height]

""" 
    Displays detected and classified image from a video frame
"""
if __name__ == '__main__':

    video_path = "D:\\GTSRB Dataset\\classifier_new\\video_input\\munich.mp4"
    output_path = "D:\\GTSRB Dataset\\classifier_new\\video_output"
    # model_name = "D:\\GTSRB Dataset\\classifier_new\\trafficsignnet.model\\20211217-230518" #original io model
    # model_name = "D:\\GTSRB Dataset\\classifier_new\\trafficsignnet.model\\20220116-062159" #cv2 model
    # model_name = "D:\\GTSRB Dataset\\classifier_new\\trafficsignnet.model\\20220116-090355"  #new model
    model_name = "D:\\GTSRB Dataset\\classifier_new\\trafficsignnet.model\\20220116-111747"  #new model v2
    
    path = os.path.join(output_path,video_path.split("\\")[-1])
    os.makedirs(path, exist_ok=True)

    import time
    start_time = time.time()
    frame_count = 0
    tfps = 30
    cap = cv2.VideoCapture(video_path)
    curr_frame = 0
    i=0
    print ("Set 1: ", str(round(time.time()-start_time, 2)))
    start_time = time.time()

    print("[INFO] loading model...")
    model = load_model(model_name)
    print ("Set 2: ", str(round(time.time()-start_time, 2)))
    start_time = time.time()

    #yolo setup
    net = cv2.dnn.readNet("yolov4-tiny_training_last.weights", "yolov4-tiny_training.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    confidence_threshold = 0.5
    print ("Set 3: ", str(round(time.time()-start_time, 2)))
    
    # load the label names
    labelNames = open("signnames.csv").read().strip().split("\n")[1:]
    labelNames = [l.split(",")[1] for l in labelNames]

    while(True):
        # start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        elif np.count_nonzero(np.array(frame.shape))<3:
            continue
        i+=1
        # print ("Set 4: ", str(round(time.time()-start_time, 2)))
        if i%4 == 0:
            try:
                start_time = time.time()

                curr_frame +=1
                image_path = 'captured_frames/frame'+str(curr_frame)+'.jpg'
                cv2.imwrite('captured_frames/frame'+str(curr_frame)+'.jpg',frame)
                #         img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # print ("Set 3: ", str(round(time.time(), 2)))
                confidences = []
                boxes = []
                rois = []
                labels = []
                bbs=[]


                #forward pass yolo
                image = cv2.imread(image_path)
                blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                net.setInput(blob)
                outs = net.forward(output_layers)

                #reading again since classfier performing better on skimage
                image=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
              
                for out in outs:
                    for detection in out:
                        confidence = np.max(detection[5:])

                        if confidence > confidence_threshold:
                            roi, box = crop_roi(image,detection)
                            confidences.append(float(confidence))
                            rois.append(roi)
                            boxes.append(box)

                #adjust overlaps
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

                for i,roi in enumerate(rois):
                    if i in indexes:
                        label = predict_label(model, roi, labelNames)
                        labels.append(label)
                        bbs.append(boxes[i])

                image = detected_image(image_path, bbs, labels)    
                io.imsave(path+"\\frame"+str(curr_frame)+'.jpg',image)
                elapsed_time = time.time() - start_time
                fps = 1/elapsed_time
                print ("frame", curr_frame)
                print ("fps: ", str(round(fps, 2)))
            except ValueError:
                continue
    cap.release()
    cv2.destroyAllWindows()