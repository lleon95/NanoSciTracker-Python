#!/usr/bin/env python3

import argparse
import copy
import cv2 as cv
import time

import detector
import tracker
import drawutils

def main(args):
  # Open the video
  cap = cv.VideoCapture(args.input)
  counter = 0
  trackers = []
  
  
  while(cap.isOpened()):
    # Grab the frame
    ret, big_frame = cap.read()
    if not ret:
      break
    
    subtrackers = []
    frame = cv.resize(big_frame, (640,480))
    
    # Grayscale
    gray_detect = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray_track = copy.deepcopy(gray_detect)
    gray_detect2 = copy.deepcopy(gray_detect)
    gray_detect3 = copy.deepcopy(gray_detect)
    
    # Detection - Top detection
    detection_bbs = detector.detect(gray_detect)
    
    # Tracking - Top tracking
    if args.sample_tracking == counter:
      trackers = tracker.deployTrackers(frame, detection_bbs, trackers)
    else:
      tracker.updateTrackers(frame, trackers)
    
    # Draw on demand
    if args.draw_detection:
      detections_frame = copy.deepcopy(frame)
      detections_frame = drawutils.draw_detections(detections_frame, detection_bbs)     
      detections_frame = cv.resize(detections_frame, (320,240))
      cv.imshow("Detection", detections_frame)
      
    if args.draw_tracking:
      tracking_frame = copy.deepcopy(frame)
      tracking_frame = drawutils.draw_trackers(tracking_frame, trackers)
      tracking_frame = cv.resize(tracking_frame, (320,240))
      cv.imshow("Tracking", tracking_frame)
      
    output_frame = cv.resize(frame, (320,240))
    cv.imshow("Original", output_frame)  
    counter += 1
    cv.waitKey(1)

    time.sleep(0.5)
    
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
                        help='Determine how many samples needed to update the tracker',
                        default=20)
  
  args = parser.parse_args()
  
  # Execute the main
  main(args)