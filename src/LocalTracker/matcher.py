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

from shapely.geometry import Polygon
import numpy as np

def compute_area(bbox):
    p1, p2 = bbox
    w = p2[0] - p1[0]
    h = p2[1] - p1[1]
    return w * h

def calculate_cd(b1, b2):
    '''
    Computes the distance of the centers

    Params:
    * b1, b2: bounding boxes

    Returns:
    * L2 distance
    '''
    p1_1, p2_1 = b1
    p1_2, p2_2 = b2

    c1 = ((p1_1[0] + p2_1[0])/2, (p1_1[1] + p2_1[1])/2)
    c2 = ((p1_2[0] + p2_2[0])/2, (p1_2[1] + p2_2[1])/2)

    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]

    d = (dx * dx) + (dy * dy)
    return np.sqrt(d)

def calculate_iom(b1, b2):
    '''
    Computes the IoM of the bounding boxes

    Params:
    * b1, b2: bounding boxes

    Returns:
    * iou
    '''
    p1_1, p2_1 = b1
    p1_2, p2_2 = b2

    # Adapt format
    box_1 = [[p1_1[0], p1_1[1]], [p2_1[0], p1_1[1]], [p1_1[0], p2_1[1]],
            [p2_1[0], p2_1[1]]]
    box_2 = [[p1_2[0], p1_2[1]], [p2_2[0], p1_2[1]], [p1_2[0], p2_2[1]], 
            [p2_2[0], p2_2[1]]]

    # Create the polygons
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)

    poly_1 = poly_1.buffer(0)
    poly_2 = poly_2.buffer(0)

    # Argmin of the union to compute the overlap in case of containing
    union = min(poly_1.area, poly_2.area)

    if union == 0:
        return 0

    iom = poly_1.intersection(poly_2).area / union
    return iom


def inter_match(detection_bbs, trackers, threshold={"iom": 0.25, "cd":64}):
    '''
    Matches the bounding boxes

    Parameters:
    * detection_bbs: New detections
    * tracking_bbs: Trackers
    * threshold: Threshold to discriminate new detections

    Return:
    * Valid new detections
    '''
    valid = []
    for detection in detection_bbs:
        # Compute the area of the detection
        p1_1, p2_1 = detection
        a1 = compute_area(detection)
        overlap = False

        for tracker in trackers:
            # Look if there is an intersection
            iou = calculate_iom(detection, tracker.roi)
            if iou > threshold["iom"]:
                overlap = True

            # Look if the new detection is close to a tracker
            cd = calculate_cd(detection, tracker.roi)
            if cd < threshold["cd"]:
                overlap = True

            # If it's overlapping and it's not death nothing to do
            if overlap:
                break

        if not overlap:
            valid.append(detection)
    return valid
