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

import numpy as np
import copy
import cv2 as cv
from skimage.feature import hog

from drawutils import crop_roi

def computeTrackerRoi(roi):
    x1 = roi[0][0]
    y1 = roi[0][1]
    x2 = roi[1][0]
    y2 = roi[1][1]
    return (x1, y1, x2 - x1, y2 - y1)

def computeCenterRoi(roi):
    x1 = roi[0][0]
    y1 = roi[0][1]
    x2 = roi[1][0]
    y2 = roi[1][1]
    # Compute center
    xc = (x2 + x1)/2.
    yc = (y2 + y1)/2.
    return (xc, yc)

def computeDirection(dx, dy):
    return np.arctan2(dy, dx)

def compute_histogram(cropped):
    return cv.calcHist([cropped],[0],None,[96],[64,256])

def compute_hog(gray, tracker):
    x1 = tracker.roi[0][0]
    y1 = tracker.roi[0][1]
    x2 = tracker.roi[1][0]
    y2 = tracker.roi[1][1]
    padding = tracker.hog_padding
    area = tracker.hog_area
    hog_ = None

    if x1 > padding[0] and y1 > padding[1] and \
        x2 < (area[0] - padding[0]) and y2 < (area[1] - padding[1]):
        hog_ = hog(gray, orientations=9, pixels_per_cell=(8, 8),
            cells_per_block=(1, 1), feature_vector=True)
    return hog_

class SpeedFeature():
    def __init__(self, mmp):
        # Feature - speed
        self.speed = -1
        self.speed_vector = list([])
        self.speed_counter = 0
        self.mobile_mean_param = mmp

    def _compute(self):
        if self.speed_counter < self.mobile_mean_param:
            return -1
        
        self.speed = np.mean(np.gradient(np.array(self.speed_vector), 1))
        return self.speed

    def add_sample(self, position):
        self.speed_counter += 1
        if len(self.speed_vector) > self.mobile_mean_param:
            self.speed_vector.pop(0)
        self.speed_vector.append(position)
        return self._compute()

class Tracker:
    def __init__(self, colour, timeout=5):
        self.tracker = cv.TrackerKCF_create()
        self.colour = colour
        self.roi = None
        self.timeout = timeout

        # Hyperparams
        self.speed_bins = 30
        self.histo_lr = 0.1
        self.hog_lr = 0.1
        self.hog_padding = (64, 64)
        self.hog_area = (640,480)

        # Features
        self.speed = (SpeedFeature(self.speed_bins), \
            SpeedFeature(self.speed_bins)) # (x,y)
        self.direction = 0
        self.histogram = None
        self.hog = None
        
    def init(self, frame, roi):
        self.roi = roi
        tracker_roi = computeTrackerRoi(roi)

        # Initialise some features
        cropped = crop_roi(frame, roi)
        gray = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
        self.histogram = compute_histogram(gray)
        self.hog = compute_hog(gray, self)

        # Initialise tracker
        return self.tracker.init(frame, tracker_roi)
    
    def _update_speed(self):
        xc, yc = computeCenterRoi(self.roi)
        dx, dy = self.speed
        dx_v = dx.add_sample(xc)
        dy_v = dy.add_sample(yc)
        if dx_v != 0.:
            self.direction = computeDirection(dx_v, dy_v)

    def _update_histogram(self, gray_roi):
        hist1 = compute_histogram(gray_roi)
        self.histogram = self.histogram * (1 - self.histo_lr) + \
            self.histo_lr * hist1

    def _update_hog(self, gray_roi):
        hog1 = compute_hog(gray_roi, self)
        if not hog1 is None:
            if self.hog is None:
                self.hog = hog1
            else:
                self.hog = self.hog * (1 - self.hog_lr) + \
                    self.hog_lr * hog1
         
    def update(self, frame):
        ok, bbox = self.tracker.update(frame)
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

        if ok:
            self.roi = (p1, p2)

        # Crop and grayscale
        cropped = crop_roi(frame, self.roi)
        gray = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)

        # Update features
        self._update_speed()
        self._update_histogram(gray)
        self._update_hog(gray)

        if self.timeout == 0:
            return False
        if not ok:
            self.timeout -= 1
        return True
    
def updateTrackers(frame, trackers):
    i = 0
    length = len(trackers)
    
    while i < length:
        state = trackers[i].update(frame)
        if not state:
            length -= 1
            del trackers[i]
        else:
            i += 1
    return trackers

def deployTrackers(colour, bb_list, trackers):
    for i in bb_list:
        tracker = Tracker((0,255,0))
        tracker.init(colour, i)
        trackers.append(tracker)
    return trackers

def retrieveBBs(trackers):
    bounding_boxes = []
    for tracker in trackers:
        bounding_boxes.append(tracker.roi)
    return bounding_boxes
    