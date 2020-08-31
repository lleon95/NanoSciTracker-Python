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
from drawutils import computeCenterRoi

from features.feature import Feature

class SpeedFeature():
    def __init__(self, mmp):
        # Feature - speed
        self.speed = -1
        self.speed_vector = list([])
        self.speed_counter = 0
        self.mobile_mean_param = mmp

    def _compute(self):
        if self.speed_counter < self.mobile_mean_param:
            return -1
        
        self.speed = np.mean(np.gradient(np.array(self.speed_vector), 1))
        return self.speed

    def add_sample(self, position):
        self.speed_counter += 1
        if len(self.speed_vector) > self.mobile_mean_param:
            self.speed_vector.pop(0)
        self.speed_vector.append(position)
        return self._compute()

class Velocity(Feature):
    def __init__(self, mmp=30):
        super().__init__()
        # Feature - speed
        self.speed = None
        self.direction = None
        self.position = None
        self.mmp = mmp

    def _computeDirection(self, dx, dy):
        return np.arctan2(dy, dx)

    def update(self, roi):
        if self.speed is None:
            return False

        self.position = computeCenterRoi(roi)
        xc, yc = self.position
        dx, dy = self.speed
        dx_v = dx.add_sample(xc)
        dy_v = dy.add_sample(yc)
        if dx_v != 0.:
            self.direction = self._computeDirection(dx_v, dy_v)

    def initialise(self, roi):
        self.speed  = [SpeedFeature(self.mmp), SpeedFeature(self.mmp)]
        self.direction = [0,0]
        self.position = computeCenterRoi(roi)
        return True

    def predict(self):
        return True