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
import cv2 as cv
import sys
import time

sys.path.append("../")
sys.path.append("../LocalTracker/")

import Playground.generator as Generator
import GlobalTracker.scene as Scene

def build_rois(roi_size, overlapping):
  h, w = roi_size
  h_p = h - overlapping
  w_p = w - overlapping

  ROIS = [((0,w),(0,h)),
       ((w_p,w_p + w),(h_p, h_p + h)),
       ((0,w),(h_p,h_p+h)),
       ((w_p,w_p+w),(0,h))]
  return ROIS

def generate_scenes(rois, overlapping=0, sampling_rate=3):
  scenes = []
  for roi in rois:
    scenes.append(Scene.Scene(ROI=roi, overlap=overlapping, \
      detection_sampling=sampling_rate))
  return scenes

def main(args):
  # Generate the world
  my_world = Generator.World(playground_size=args.world_size,
                          mitosis=args.mitosis_rate,
                          instances=args.number_of_instances,
                          frames=args.frames)
  
  # Generate scenes
  rois = build_rois(args.scene_size, args.overlapping)
  scenes = generate_scenes(rois, args.overlapping, \
    args.sampling_rate_detection)

  # Run the simulation
  while(my_world.update()):
    drawing = my_world.draw()
    for i in range(len(rois)):
      # Refresh scene
      scene_frame = Generator.getFrame(rois[i], drawing)
      scenes[i].update(scene_frame)
      drawing_scene = scenes[i].draw(scene_frame)
      # Display
      cv.imshow("Scene " + str(i+1), drawing_scene)
      cv.waitKey(1)
    time.sleep(args.delay_player)

if __name__ == "__main__":
  # Handle the arguments
  parser = argparse.ArgumentParser(description='Performs the local tracking')
  
  parser.add_argument('--world_size', type=object,
                      help='Size of the world (h, w, chans)', default=(1200,1400,3))
  parser.add_argument('--scene_size', type=object,
                      help='Size of the scene within the world (h, w)', default=(480, 640))
  parser.add_argument('--overlapping', type=int,
                      help='Overlapping of the scene in pixels', default=20)
  parser.add_argument('--number_of_instances', type=int,
                      help='Number of initial cells',
                      default=15)
  parser.add_argument('--frames', type=int,
                      help='Number of frames', default=450)
  parser.add_argument('--mitosis_rate', type=float,
                      help='Mitosis rate in probability',
                      default=0.0006)
  parser.add_argument('--delay_player', type=float,
                      help='Timer delay in seconds',
                      default=0.01)
  parser.add_argument('--sampling_rate_detection', type=float,
                      help='Decimation of the detection',
                      default=3)
  
  args = parser.parse_args()
  main(args)
