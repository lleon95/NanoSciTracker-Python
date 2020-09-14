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


def destitch(img, recorders, frames_stitching, localsize):
    for i in range(frames_stitching[0]):
        for j in range(frames_stitching[1]):
            index = i + frames_stitching[0] * j
            # Crop
            y0 = j * localsize[1]
            y1 = (j + 1) * localsize[1]
            x0 = i * localsize[0]
            x1 = (i + 1) * localsize[0]
            frame = img[y0:y1, x0:x1]
            recorders[index].append(frame)


def load(path="../data/mcherry", n=4, resizeTo=(640, 480), k=7):
    """
    Get a numpy array with the shape (n, m, h, w), where n is the number of scenes,
    m is the number of frames, h is the image height and w is the image width
    """
    # Sizes
    size = (2560, 1920)
    local_size = (1280, 960)

    # Open video
    caps = []
    for i in range(n):
        caps.append(cv.VideoCapture(path + "/mcherry"+str(i)+".mp4"))

    # Number of images within the stitching
    frames_stitching = (2, 2)

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
                print(".", end='')
            else:
                print("-", end='')
                break
            if cnt % 10 == 0:
                sys.stdout.flush()
            cnt += 1
        print("Loaded video " + str(i))

    for i in range(n):
        caps[i].release()

    # Convert into numpy array
    return np.array(data), [0, 1, 2, 3]
