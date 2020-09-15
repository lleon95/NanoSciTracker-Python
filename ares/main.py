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

sys.path.append("../src")
sys.path.append("../src/GlobalTracker")
sys.path.append("../src/LocalTracker")

import GlobalTracker.world as World
import GlobalTracker.utils as Utils
import mcherry as Dataset

## --------- Parameters ------------
SCENE_SIZE = (480, 640)
WORLD_SIZE = (960, 1280)

def main(args):
  global SCENE_SIZE, WORLD_SIZE

  # Retrieve the dataset
  h, w = SCENE_SIZE
  print("Loading data...")
  dataset, order = Dataset.load(args.dataset, resizeTo=(w, h))
  print("Loaded: ", dataset.shape)
  n_frames = dataset.shape[1]

  # Attach world tracker
  tracking_world = World.World()
  
  # Generate scenes
  rois = Utils.build_rois(SCENE_SIZE, args.overlapping)
  tracking_world.spawn_scenes(rois, args.overlapping, \
    args.sampling_rate_detection)

  if args.record:
    fourcc = cv.VideoWriter_fourcc(*'MP4V')
    record = cv.VideoWriter('./video.mp4',fourcc, 25, WORLD_SIZE, True)

  # Run the simulation
  for frame_idx in range(n_frames):
    frames = []
    for i in range(len(rois)):
      # Refresh scene
      frames.append(cv.cvtColor(dataset[i][frame_idx], cv.COLOR_GRAY2BGR))

    # Update scenes
    tracking_world.update_trackers(frames)

    # Draw rectange overlay to determine where are the ROIS
    drawing = Utils.naive_stitch(frames, WORLD_SIZE, SCENE_SIZE, order)
    Utils.draw_roi(drawing, rois)

    # Label the world objects
    world_labeled = tracking_world.draw_trackers(drawing)
    cv.imshow("World", world_labeled)

    if args.record:
      record.write(world_labeled)
    cv.waitKey(1)

    time.sleep(args.delay_player)
  
  if args.record:
    record.release()


if __name__ == "__main__":
  # Handle the arguments
  parser = argparse.ArgumentParser(description='Performs the local tracking')
  parser.add_argument('--dataset', type=str,
                      help='Choose the dataset', default='../data/mcherry')
  parser.add_argument('--overlapping', type=int,
                      help='Overlapping of the scene in pixels', default=10)
  parser.add_argument('--frames', type=int,
                      help='Number of frames', default=450)
  parser.add_argument('--delay_player', type=float,
                      help='Timer delay in seconds',
                      default=0.01)
  parser.add_argument('--sampling_rate_detection', type=float,
                      help='Decimation of the detection',
                      default=3)
  parser.add_argument('--record', type=bool, help='Enable video recording',
                      default=False)
  parser.add_argument('--framerate', type=float,
                      help='In case of recording, the framerate',
                      default=25)
  
  args = parser.parse_args()
  main(args)
