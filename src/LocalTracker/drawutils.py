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
def place_id(tracker, frame, p2=None):
    font = cv.FONT_HERSHEY_SIMPLEX
    if p2 is None:
        p1, p2 = tracker.roi
    frame = cv.putText(frame, str(tracker.label['id']), p2, font, 1, \
        (255,255,255), 2)
    return frame

def draw_tracker(frame, tracker, colour=None, offset=(0,0)):
    # Extract the roi
    p1, p2 = tracker.roi
    if colour is None:
        draw_colour = tracker.colour
    else:
        draw_colour = colour
    # Add offset
    if not tracker.roi_offset is None:
        offset = tracker.roi_offset
    p1 = (p1[0] + offset[0], p1[1] + offset[1])
    p2 = (p2[0] + offset[0], p2[1] + offset[1])
    # Draw on the frame
    frame = cv.rectangle(frame, p1, p2, draw_colour, 3, 1)
    if not tracker.label is None:
        frame = place_id(tracker, frame, p2)
    return frame
    
def draw_trackers(frame, trackers, colour=None, offset=(0,0)):
    for i in trackers:
        frame = draw_tracker(frame, i, colour, offset)
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

def computeCenterRoi(roi):
    x1 = roi[0][0]
    y1 = roi[0][1]
    x2 = roi[1][0]
    y2 = roi[1][1]
    # Compute center
    xc = (x2 + x1)/2.
    yc = (y2 + y1)/2.
    return (xc, yc)

'''
World utils
'''
def place_text(frame, text, position):
    font = cv.FONT_HERSHEY_SIMPLEX
    frame = cv.putText(frame, text, position, font, 1, \
        (255,255,255), 2)
    return frame
