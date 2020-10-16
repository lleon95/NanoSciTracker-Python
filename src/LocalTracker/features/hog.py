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
import numpy as np
from skimage.feature import hog

from features.feature import Feature

class Hog(Feature):
    def __init__(self, padding=(64, 64), area=(640,480), lr=0.2):
        super().__init__()
        self.hog = None

        # Hyper-parameters
        self.padding = padding
        self.area = area
        self.orientations = 17
        self.cells_per_block = (1,1)
        self.lr = lr

    def _compute_hog(self, gray, roi):
        x1 = roi[0][0]
        y1 = roi[0][1]
        x2 = roi[1][0]
        y2 = roi[1][1]

        padding = self.padding
        area = self.area
        
        hog_ = hog(gray, orientations=self.orientations, pixels_per_cell=(y2-y1, x2-x1),
            cells_per_block=self.cells_per_block, feature_vector=True)

        if len(hog_) == 0:
            hog_ = None
        return hog_

    def update(self, gray_roi, roi):
        hog1 = self._compute_hog(gray_roi, roi)
        if not hog1 is None:
            if self.hog is None:
                self.hog = hog1
            else:
                self.hog = self.hog * (1 - self.lr) + \
                    self.lr * hog1
        else:
            return False
        return True

    def initialise(self, gray, roi):
        self.hog = self._compute_hog(gray, roi)
        return self.hog

    def predict(self):
        return copy.deepcopy(self.hog)

    def compare(self, hog2):
        # Applying Bhattacharyya to it
        X = self.predict().flatten()
        Y = hog2.predict().flatten()
        # Normalise to 1
        X /= np.sum(X)
        Y /= np.sum(Y)
        # Compute the Bhattacharyya
        bc = np.sum(np.sqrt(X * Y))
        return np.array([bc])
