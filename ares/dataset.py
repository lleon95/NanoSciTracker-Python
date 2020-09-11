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

import Utils.tiff as TiffUtils


def sortKey(name):
    file = name.split("/")[-1]
    splitted = file.split(".")
    splitted = splitted[0].split("_")
    X = int(splitted[1])
    return X


def load(path="../data/ctrl6_72h", n=4, resizeTo=(640, 480)):
    """
    Get a numpy array with the shape (n, m, h, w), where n is the number of scenes,
    m is the number of frames, h is the image height and w is the image width
    """
    normalisation = 2048
    black_offset = 64

    files = glob.glob(path + "/*.tif")
    files.sort(key=sortKey)

    # Prepare array:
    data = []
    for i in range(n):
        data.append(list([]))

    for i in files:
        file = i.split("/")[-1]
        splitted = file.split(".")
        splitted = splitted[0].split("_")
        # Format : image3_X_Y.tiff -> ['image3', X, Y]

        Y = int(splitted[2])
        if Y == 1:
            continue
        X = int(splitted[1])
        mod = X % n

        # Open image - Normalised to 2048
        tiff = TiffUtils.tiff12_open(i, normalisation)
        tiff = cv.resize(tiff, resizeTo)

        # Black normalisation
        tiff += black_offset

        # Append data
        data[mod].append(tiff)

    # Convert into numpy array
    return np.array(data), [i for i in range(n)]
