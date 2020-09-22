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

import cv2 as cv
import glob
import numpy as np
import sys

sys.path.append("../src/")

SCENE_SIZE = (960, 1280)
SCENE_SIZE_SW = (1280, 960)
WORLD_SIZE = (960, 1280)


def get_rois(roi_size, overlapping):
    h, w = roi_size
    return [((0, w), (0, h))]


def load(settings=None, n=1, resizeTo=SCENE_SIZE_SW, k=7):
    """
    Get a numpy array with the shape (n, m, h, w), where n is the number of scenes,
    m is the number of frames, h is the image height and w is the image width
    """
    if settings is None:
        raise RuntimeError("Error: settings cannot be None in dataset loader")
    
    # Sizes
    H, W = settings.set_if_defined("original_size", [1920, 2560])
    size = local_size = (W, H)

    # Open video
    caps = []
    n = settings.set_if_defined("scenes", n)
    path = "../"
    path += settings.set_if_defined("file_path", "data/mcherry")
    prefix = settings.set_if_defined("file_prefix", "mcherry")
    suffix = settings.set_if_defined("file_suffix", ".avi")
    is_enumered = settings.set_if_defined("file_enumered", False)

    for i in range(n):
        if is_enumered:
            file = path + "/" + prefix + str(i) + suffix
        else:
            file = path + "/" + prefix + suffix
        caps.append(cv.VideoCapture(file))

    # Number of images within the stitching
    frames_stitching = settings.set_if_defined("stitching", [1,1])

    # Prepare array:
    data = []
    for i in range(n):
        data.append(list([]))

    # Frame
    for i in range(n):
        print("Video " + str(i))
        cnt = 0
        while caps[i].isOpened():
            ret, frame = caps[i].read()
            if ret == True:
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                frame = cv.resize(frame, resizeTo)
                data[i].append(frame)
                print(".", end="")
            else:
                print("-", end="")
                break
            if cnt % 10 == 0:
                sys.stdout.flush()
            cnt += 1
        print("Loaded video " + str(i))

    for i in range(n):
        caps[i].release()

    # Convert into numpy array
    return np.array(data), settings.set_if_defined("stitching_order", [0])
