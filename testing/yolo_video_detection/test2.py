# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 11:02:37 2021

@author: devji
"""


# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 00:54:22 2021

@author: devji
"""
import cv2

vid_capture = cv2.VideoCapture('video_data/2.mp4')

if (vid_capture.isOpened() == False):
    print("Error opening the video file")
else:
    fps = vid_capture.get(5)
    frame_count = vid_capture.get(7)
    
while(vid_capture.isOpened()):
    ret, frame = vid_capture.read()
    if ret == True:
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(20)
        
        if key == ord('q'):
            break;
    else:
        break
    
vid_capture.release()
cv2.destroyAllWindows()
