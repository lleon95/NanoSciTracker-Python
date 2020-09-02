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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import cv2 as cv
import seaborn as sns
import copy

import random

# -----------------------------------------------------------------------------
# Global variables
# -----------------------------------------------------------------------------
# Number of moving instances
INSTANCES=15
# World size - where the cells will live canvas
PLAYGROUND_SIZE=(1200, 1400, 3) # Y,X, Colour
# Would background colour
PLAYGROUND_COLOUR = [0,0,0]
# ROIs to extract from the world
ROIS = [((0,640),(0,480)),
       ((620,620+640),(460,460+480)),
       ((0,640),(460,460+480)),
       ((620,620+640),(0,480))]
# Speeds
MIN_SPEED=1
MAX_SPEED=5
LEARNING_RATE=0.05
# SIZE
MIN_SIZE = 15
MAX_SIZE = 35
# Initial position margins
MIN_X = 100
MAX_X = PLAYGROUND_SIZE[0]-100
MIN_Y = 100
MAX_Y = PLAYGROUND_SIZE[1]-100
# Timing
FRAME_RATE=30
FRAMES=15*30

# -----------------------------------------------------------------------------
# Cell class
# Smallest particle in the world
# -----------------------------------------------------------------------------
class Cell:
    '''
    Instance which will populate the world
    '''
    def __init__(self, colour, speed_limits=(1,5), \
      boundary_limits=((100,1400-100), (100,1200-100)), \
      size_limits=(15,35), lr=0.05):
        # Set the limits
        self.min_x = boundary_limits[0][0]
        self.max_x = boundary_limits[0][1]
        self.min_y = boundary_limits[1][0]
        self.max_y = boundary_limits[1][1]
        self.min_speed = speed_limits[0]
        self.max_speed = speed_limits[1]
        self.min_size = size_limits[0]
        self.max_size = size_limits[1]
        self.lr = 0.05
        # Initial position
        self.x = self.min_x + (self.max_x - self.min_x) * random.random()
        self.y = self.min_y + (self.max_y - self.min_y) * random.random()
        # Initial speed
        self.dx = (2 * (self.max_speed) * random.random() - self.max_speed)
        self.dy = (2 * (self.max_speed) * random.random() - self.max_speed)
        # My shape
        self.size = self.min_size + (self.max_size - self.min_size) * \
          random.random()
        self.colour = colour
        # Active flag - inactive when it goes out of scene
        self.active = True
        
    def update(self):
        # Update position
        self.x += self.dx
        self.y += self.dy
        # Update speed
        dx = self.lr * (2 * (self.max_speed) * random.random() - \
          self.max_speed)
        dy = self.lr * (2 * (self.max_speed) * random.random() - \
          self.max_speed)
        if abs(self.dx + dx) < self.max_speed:
            self.dx += dx
        if abs(self.dy + dy) < self.max_speed:
            self.dy += dy
        # Update size
        size = self.lr * (2 * self.max_size * random.random() - \
          self.max_size) * 0.1
        if self.size + size < self.max_size and self.size + \
          size >= self.min_size:
            self.size += size
            
        # Corner cases
        if self.x < -self.max_size:
            self.dx = (self.max_speed + self.min_speed) * 0.5
        if self.y < -self.max_size:
            self.dy = (self.max_speed + self.min_speed) * 0.5
        if self.x > self.max_size + self.max_x + self.min_x:
            self.dx = -(self.max_speed + self.min_speed) * 0.5
        if self.y > self.max_size + self.max_y + self.min_y:
            self.dy = -(self.max_speed + self.min_speed) * 0.5
        
    def isActive(self):
        return self.active
    
    def deactivate(self):
        self.active = False
        
# -----------------------------------------------------------------------------
# World class
# Group of cells on a canvas
# -----------------------------------------------------------------------------   
class World:
    '''
    The world canvas - black
    '''
    def __init__(self, playground_size=(1200, 1400, 3), \
      playground_bg=[0,0,0], frames=15*30, instances=25, roi_padding=10, \
      mitosis=0.0006):
        self.initial_seed = instances * 2 // 3 # 66%
        self.canvas = np.full(playground_size, playground_bg, dtype=np.uint8)
        self.active_instances = []
        self.missing_life = frames
        self.playground_size = playground_size
        self.playgound_colour = playground_bg
        self.instances = instances
        self.roi_padding = roi_padding
        self.last_instance_idx = self.initial_seed
        self.mitosis_rate = mitosis 
        self.total_instances = instances
        
        # Create instances
        colour_palette = sns.color_palette("hls", instances)
        for c in range(self.initial_seed):
            i = colour_palette[c]
            colour = (i[0] * 255, i[1] * 255, i[2] * 255)
            self.active_instances.append(Cell(colour))
            
    def getROI(self, i):
        '''
        Gets the ROI for an instance
        '''
        x1 = int(i.x - i.size - self.roi_padding)
        x2 = int(i.x + i.size + self.roi_padding)
        y1 = int(i.y - i.size - self.roi_padding)
        y2 = int(i.y + i.size + self.roi_padding)
        return ((x1,y1),(x2,y2))
            
    def getBBs(self):
        '''
        Get the current bounding boxes of the instances if they are valid
        '''
        bbs = list([])
        for i in self.active_instances:
            if i.x < self.playground_size[1] and i.y < self.playground_size[0] \
              and i.x >= 0 and i.y >= 0:
                x1 = int(i.x - i.size - self.roi_padding)
                x2 = int(i.x + i.size + self.roi_padding)
                y1 = int(i.y - i.size - self.roi_padding)
                y2 = int(i.y + i.size + self.roi_padding)
                bbs.append(((x1,y1),(x2,y2)))
        return bbs
        
    def draw(self):
        '''
        Draws all the instances on the canvas (or redraw if invoked several 
        times)
        '''
        self.canvas = np.full(self.playground_size, self.playgound_colour, \
          dtype=np.uint8)
        for i in self.active_instances:
            if i.x < self.playground_size[1] and i.y < self.playground_size[0] \
              and i.x >= 0 and i.y >= 0:
                self.canvas = cv.circle(self.canvas, (int(i.x), int(i.y)), \
                  int(i.size), i.colour, int(-1))
                roi = self.getROI(i)
                self.canvas = cv.rectangle(self.canvas, roi[0], roi[1], \
                  i.colour, 1) 
        return self.canvas
    
    def update(self):
        '''
        Run the per-instance update
        '''
        current_len = len(self.active_instances)
        for j in range(current_len):
            i = self.active_instances[j]
            if i.isActive():
                # Update the cell
                i.update()
                # Should it do mitosis
                mit_prob = random.random() < self.mitosis_rate
                if mit_prob and self.last_instance_idx < self.total_instances:
                    child = copy.deepcopy(i)
                    self.active_instances.append(child)
                    self.last_instance_idx += 1
                    print("Mitosis event registered")
                    
        self.missing_life -= 1
        if self.missing_life == 0:
            return False
        else:
            return True


# -----------------------------------------------------------------------------
# Testing resources
# -----------------------------------------------------------------------------
# Create the video capture
def getSize(ROI):
    y = ROI[1][0]
    h = ROI[1][1] - y
    x = ROI[0][0]
    w = ROI[0][1] - x
    return (w, h)

# Get frame
def getFrame(ROI, drawing):
    y = ROI[1][0]
    h = ROI[1][1]
    x = ROI[0][0]
    w = ROI[0][1]

    return drawing[y:h,x:w]

if __name__ == "__main__":
    # Create capturers
    fourcc = cv.VideoWriter_fourcc(*'H264')
    outputs = []
    play_out = cv.VideoWriter('playground.mp4',fourcc, FRAME_RATE, \
      (PLAYGROUND_SIZE[1], PLAYGROUND_SIZE[0]), True)
    for i in range(len(ROIS)):
        outputs.append(cv.VideoWriter('playground'+str(i)+'.mp4',fourcc, \
          FRAME_RATE, getSize(ROIS[i]), True))

    # Create world
    my_world = World()

    # Run the simulation
    while(my_world.update()):
        drawing = my_world.draw()
        play_out.write(drawing)
        for i in range(len(ROIS)):
            outputs[i].write(getFrame(ROIS[i], drawing))

    # Release
    for i in outputs:
        i.release()
    play_out.release()
