#!/usr/bin/env python3

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

import argparse
import copy
import cv2 as cv
import time
import matplotlib.pyplot as plt

import detector
import tracker
import drawutils
import matcher

def main(args):
  # Open the video
  cap = cv.VideoCapture(args.input)
  counter = 0
  trackers = []
  
  plt.ion()

  detection_roi = (40, 40, 600, 440)
  
  while(cap.isOpened()):
    # Grab the frame
    ret, big_frame = cap.read()
    if not ret:
      break
    
    detection_bbs = []
    new_detections = []
    frame = cv.resize(big_frame, (640,480))
    
    # Grayscale
    gray_detect = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray_track = copy.deepcopy(gray_detect)
    gray_detect2 = copy.deepcopy(gray_detect)
    gray_detect3 = copy.deepcopy(gray_detect)
    
    # Detection - Refresh tracking
    if counter % args.sample_detection:
      detection_bbs = detector.detect(gray_detect, ROI=detection_roi)
      new_detections = matcher.inter_match(detection_bbs, tracking_bbs)
      trackers = tracker.deployTrackers(frame, new_detections, trackers)
      
    # Update the trackers
    tracker.updateTrackers(frame, trackers, ROI=detection_roi)
    tracking_bbs = tracker.retrieveBBs(trackers)

    if len(trackers) > 5:
      plt.clf()
      print("dx:", round(trackers[0].speed[0].speed, 2), \
        "dy:", round(trackers[0].speed[1].speed, 2), \
          "angle:", round(trackers[0].direction * 180 / 3.1416, 2))
      trackers[3].colour = (255,0,0)
      for i in range(3):
        if not trackers[i].hog is None:
          plt.plot(trackers[i].hog, '*')
      plt.draw()
      
    # Draw on demand
    if args.draw_detection:
      detection_bbs = detector.detect(gray_detect, ROI=detection_roi)
      detections_frame = copy.deepcopy(frame)
      detections_frame = drawutils.draw_detections(detections_frame, detection_bbs)
      cv.rectangle(detections_frame, (detection_roi[0], detection_roi[1], \
        detection_roi[2] - detection_roi[0], detection_roi[3] - detection_roi[1]), (255,0,0), 2)
      cv.imshow("Detection", detections_frame)
      
    if args.draw_tracking:
      tracking_frame = copy.deepcopy(frame)
      tracking_frame = drawutils.draw_trackers(tracking_frame, trackers)
      cv.rectangle(tracking_frame, (detection_roi[0], detection_roi[1], \
        detection_roi[2] - detection_roi[0], detection_roi[3] - detection_roi[1]), (255,0,0), 2)
      tracking_frame = drawutils.draw_detections(tracking_frame, new_detections, (0,0,255))
      cv.imshow("Tracking", tracking_frame)
      
    output_frame = cv.resize(frame, (320,240))
    cv.imshow("Original", output_frame)  
    counter += 1
    cv.waitKey(1)

    #time.sleep(0.5)
    
  cv.destroyAllWindows()
  
if __name__ == "__main__":
  # Handle the arguments
  parser = argparse.ArgumentParser(description='Performs the local tracking')
  
  parser.add_argument('--input', type=str,
                        help='Location for the video sequence', required=True)
  parser.add_argument('--draw_detection', type=bool,
                        help='Enable the detection bbs', default=False)
  parser.add_argument('--draw_tracking', type=bool,
                        help='Enable the tracking bbs', default=False)
  parser.add_argument('--sample_tracking', type=int,
                        help='Determine how many samples needed to purge the tracker',
                        default=30)
  parser.add_argument('--sample_detection', type=int,
                        help='Determine how many samples needed to update the detection',
                        default=3)
  
  args = parser.parse_args()
  
  # Execute the main
  main(args)