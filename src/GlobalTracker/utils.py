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

import cv2 as cv
import numpy as np

def draw_roi(world, rois):
    for roi in rois:
        p1, p2 = roi
        world = cv.rectangle(world, (p1[0], p2[0]), (p1[1], p2[1]), (128, 128, 128), 2)
    return world


def build_rois(roi_size, overlapping):
    h, w = roi_size
    h_p = h - overlapping
    w_p = w - overlapping

    ROIS = [
        ((0, w), (0, h)),
        ((w_p, w_p + w), (h_p, h_p + h)),
        ((0, w), (h_p, h_p + h)),
        ((w_p, w_p + w), (0, h)),
    ]
    return ROIS


def naive_stitch(frames, world_size, scene_size, order):
    # Create the canvas
    h, w = world_size
    world = np.zeros((h, w, 3), dtype=np.uint8)
    h, w = scene_size

    for i in order:
        x = w * (i % 2)
        y = h * (i // 2)
        world[y:y+h, x:x+w] = frames[i]

    return world
