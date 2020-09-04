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

import copy
import cv2 as cv
import numpy as np

from features.feature import Feature

class Histogram(Feature):
    def __init__(self, grayscale=False, range=[64,256], bins=[96], lr=0.1):
        super().__init__()
        self.histogram = None
        self.grayscale = grayscale

        # Hyper-parameters
        self.range = range
        self.bins = bins
        self.lr = lr

    def _compute_histogram(self, cropped):
        if self.grayscale:
            return cv.calcHist([cropped], [0], None, self.bins, self.range)
        else:
            arr = list([])
            arr.append(cv.calcHist([cropped], [0], None, self.bins, self.range))
            arr.append(cv.calcHist([cropped], [1], None, self.bins, self.range))
            arr.append(cv.calcHist([cropped], [2], None, self.bins, self.range))
            return np.array(arr) 

    def update(self, gray_roi):
        hist1 = self._compute_histogram(gray_roi)
        self.histogram = self.histogram * (1 - self.lr) + \
            self.lr * hist1
        return True

    def initialise(self, cropped):
        self.histogram = self._compute_histogram(cropped)
        return True

    def predict(self):
        return copy.deepcopy(self.histogram)

    def compare(self, histo2):
        # Applying Pearson correlation to it
        X = self.predict().flatten()
        Y = histo2.predict().flatten()
        # Compute means and variances
        r = np.corrcoef(X, Y)
        return np.array([np.abs(r).mean()])
