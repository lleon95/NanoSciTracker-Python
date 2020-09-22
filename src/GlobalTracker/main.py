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
sys.path.append("../GlobalTracker/")
sys.path.append("../Matcher/")
sys.path.append("../Utils/")

import Playground.generator as Generator
import GlobalTracker.world as World
import GlobalTracker.utils as Utils
import Utils.json_settings as Settings

def main(args):
  # Open Settings
  settings = Settings.Settings("playground.json")
  if not settings.is_valid():
    print("Error: Settings not valid")
    return

  # Generate the world
  world_size = settings.set_if_defined("world_size", args.world_size)

  my_world = Generator.World(playground_size=world_size,
                          mitosis=args.mitosis_rate,
                          instances=args.number_of_instances,
                          frames=args.frames)

  # Attach world tracker
  tracking_world = World.World(settings)
  
  # Generate scenes
  scene_size = settings.set_if_defined("scene_size", args.scene_size)
  overlapping = settings.set_if_defined("overlapping", args.overlapping)

  rois = Utils.build_rois(scene_size, overlapping)
  tracking_world.spawn_scenes(rois, overlapping, \
    args.sampling_rate_detection)

  if args.record:
    fourcc = cv.VideoWriter_fourcc(*'MP4V')
    record = cv.VideoWriter('./video.mp4',fourcc, 25, (1400,1200), True)

  # Run the simulation
  while(my_world.update()):
    drawing = my_world.draw()
    frames = []
    for i in range(len(rois)):
      # Refresh scene
      frames.append(Generator.getFrame(rois[i], drawing))

    # Update scenes
    tracking_world.update_trackers(frames)

    # Draw rectange overlay to determine where are the ROIS
    Utils.draw_roi(drawing, rois)

    # Label the world objects
    world_labeled = tracking_world.draw_trackers(drawing)
    
    if args.display:
      cv.imshow("World", world_labeled)
      cv.waitKey(1)

    if args.record:
      record.write(world_labeled)

    time.sleep(args.delay_player)
  
  if args.record:
    record.release()


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
  parser.add_argument('--no-display', help='Display analysis', dest='display', 
                      action='store_false')
  parser.add_argument('--record', help='Enable video recording',
                      dest='record', action='store_true')
  
  args = parser.parse_args()
  main(args)
