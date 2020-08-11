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

def compute_area(bbox):
    p1, p2 = bbox
    w = p2[0] - p1[0]
    h = p2[1] - p1[1]
    return w * h

def calculate_iou(b1, b2):
    '''
    Computes the IoU of the bounding boxes

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

    # Compute the union
    union = min(poly_1.area, poly_2.area)

    if union == 0:
        return 0

    iou = poly_1.intersection(poly_2).area / union
    if (iou > 0.001):
        print(iou)
    return iou


def match_overlaps(bbs, max_overlap=0.10):
    '''
    Matches the bounding boxes within a detection/tracking process. It returns
    the bounding boxes with all the ambiguities solved

    Parameters:
    * bbs: bounding boxes to analyse in

    Returns:
    * bounding boxes withou ambiguities

    Enhacement opportunity: reduce the O(n) order by an unsorted map or similar
    '''
    solve = []
    for b1 in bbs:
        # Compute the area of the b1
        p1_1, p2_1 = b1
        a1 = compute_area(b1)
        overlap = False

        # See if the points are within the area
        for b2 in bbs:
            # Skip if it's going to compare against itself
            if b1 == b2:
                continue

            # Look if there is an intersection
            iou = calculate_iou(b1, b2)
            if iou > max_overlap:
                overlap = True
                continue
            # If so
            a2 = compute_area(b2)

            # Choose the bigger - a2 will be analysed later
            
        if not overlap:
            solve.append(b1)
    return solve
            

def match_bbs(detection_bbs, tracking_bbs):
    '''
    Matches the bounding boxes
    '''
    pass