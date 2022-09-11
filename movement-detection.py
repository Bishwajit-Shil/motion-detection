#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 04:02:09 2020

@author: jitshil
"""


import cv2
import imutils
import numpy as np
import time




cam =cv2.VideoCapture('/home/jitshil/Desktop/Pantech-solution/opencv/video/1.mp4')
time.sleep(1)

firstFrame =None
area1 = 500
area2 =300


while True:
    __,img = cam.read()
    text = 'Normal'
    img = imutils.resize(img, width=1000, height= 1000)
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussianImg = cv2.GaussianBlur(grayImg, (21,21), 0)
    
    if firstFrame is None:
        firstFrame = gaussianImg
        continue
    
    imgdiff = cv2.absdiff(firstFrame, grayImg)
    thresholdimg = cv2.threshold(imgdiff, 30, 255, cv2.THRESH_BINARY)[1]
    thresholdimg = cv2.dilate(thresholdimg, None, iterations =2)
    cnts = cv2.findContours(thresholdimg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    for  c in cnts:
        (x,y,w,h) = cv2.boundingRect(c)
        # if  cv2.contourArea(contour) < 300 or cv2.contourArea(contour) > 5000 :
        if  cv2.contourArea(c) < 1000 or cv2.contourArea(c) > 2000 :
            continue
        
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        text = 'moving object detected'
    print(text)
    
    cv2.putText(img, text, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 3)
    cv2.imshow('movement detection ', img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    
cam.release()
cv2.destroyAllWindows()    