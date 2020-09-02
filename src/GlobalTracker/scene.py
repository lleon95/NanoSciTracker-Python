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

import copy
import cv2 as cv

import LocalTracker.detector as Detector
import LocalTracker.drawutils as DrawUtils
import LocalTracker.tracker as Tracker
import LocalTracker.matcher as Matcher

class Scene:
    def __init__(self, ROI=None, overlap=0, detection_sampling=3, detection_roi=None):
        # Get coordinates
        self.roi = ROI
        x, y = self.roi
        self.x, self.w = x
        self.y, self.h = y
        self.overlap = overlap

        # ROIs
        if detection_roi is None:
            self.detection_roi = (self.overlap, self.overlap, \
                self.w - self.overlap, \
                self.h - self.overlap)

        # BBs
        self.trackers = []
        self.detections = []
        self.trackings = []
        self.new_detections = []
        self.trackers_out_scene = []

        # Settings
        self.counter = 0
        self.detection_sampling = detection_sampling

    def detect(self, gray_frame):
        return Detector.detect(gray_frame, ROI=self.detection_roi)

    def track(self, colour_frame):
        Tracker.updateTrackers(colour_frame, self.trackers, \
            ROI=self.detection_roi)
        return Tracker.retrieveBBs(self.trackers)

    def update(self, colour_frame):
        gray_detect = cv.cvtColor(colour_frame, cv.COLOR_BGR2GRAY)
        # Perform detections and filter the new ones
        if self.counter % self.detection_sampling == 0:
            self.detections = self.detect(gray_detect)
            self.new_detections = Matcher.inter_match(self.detections, \
                self.trackings)
            # Deploy new trackers accordingly
            self.trackers = Tracker.deployTrackers(colour_frame, \
                self.new_detections, self.trackers)
        # Perform tracking update
        self.trackings = self.track(colour_frame)
        # Catch trackers which went out of scene
        self.trackers_out_scene = Tracker.retrieveOutScene(self.trackers)
        self.counter += 1
    
    def draw(self, colour_frame):
        '''
        Purple: New detections
        Red: Detections
        Blue: Trackers
        Light blue: Out of scene
        '''
        colour_copy = copy.deepcopy(colour_frame)
        # Draw detections
        colour_copy = DrawUtils.draw_detections(colour_copy, \
            self.new_detections, (255,0,255))
        colour_copy = DrawUtils.draw_detections(colour_copy, \
            self.detections, (0,0,255))
        # Draw trackers
        colour_copy = DrawUtils.draw_trackers(colour_copy, \
            self.trackers, (255,0,0))
        colour_copy = DrawUtils.draw_trackers(colour_copy, \
            self.trackers_out_scene, (255,255,0))
        return colour_copy
