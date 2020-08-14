# NanoSciTracker - 2020
# Author: Luis G. Leon Vega <luis@luisleon.me>
# 
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
# 
#   http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# 
# This project was sponsored by CNR-IOM
# Master in High-Performance Computing - SISSA

import cv2 as cv

'''
Painting tools for tracking
'''
def draw_tracker(frame, tracker):
    # Extract the roi
    p1, p2 = tracker.roi
    # Draw on the frame
    frame = cv.rectangle(frame, p1, p2, tracker.colour, 3, 1)
    return frame
    
def draw_trackers(frame, trackers):
    for i in trackers:
        frame = draw_tracker(frame, i)
    return frame

'''
Painting tools for detection
'''
def draw_detections(frame, bbs, colour=(255,0,0)):
    for i in bbs:
        cv.rectangle(frame, i[0], i[1], colour, 3)
    return frame

'''
Cropping tools
'''
def crop_roi(frame, roi):
    x1 = roi[0][0]
    y1 = roi[0][1]
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    x2 = roi[1][0]
    y2 = roi[1][1]
    return frame[y1:y2, x1:x2]