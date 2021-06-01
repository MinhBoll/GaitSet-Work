#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 20:21:17 2021

@author: minhdoan
"""

import os
import shutil
import cv2

src_path = 'asilla-data/preprocessed_videos_probe/'
dest_path = 'asilla-data/asilla-data'

for dirpath, dirs, files in os.walk(src_path):
    for file in files:
        print(f"Extracting video {file}")
        filename = file.split(".")[0]
        print(filename)
        if file.endswith('.mp4'):
            Dir = file.split("-")
            new_path = os.path.join(dest_path, Dir[0], Dir[1]+"-"+Dir[2], Dir[3])
            print("REAL: "+new_path)
            if (not os.path.exists(new_path)):
                print("Doesn't exists")
                os.makedirs(new_path)
                
        video = os.path.join(dirpath,file)
        print(f"Woring on {video}")
        #print("Testing"+new_path+filename+'.png')
        cap = cv2.VideoCapture(video)

        i = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #cv2.imshow('Instance Segmentation', frame)
      
            cv2.imwrite(new_path+'/'+filename+'-'+str(i).zfill(3)+'.png', frame)
            
            i+=1
            
        
        
            #shutil.move(os.path.join(dirpath, file), new_path)