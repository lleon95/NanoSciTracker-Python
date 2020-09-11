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
import numpy as np

def tiff12_open(img_path, normalisation=2048):
  '''
  Tiff 12 image opener
  img_path: path to the file
  normalisation: max value within the image
  '''
  tiff = cv.imread(img_path, -1)

  if tiff is None:
    print("No image loaded...")
    return

  # Compute the image
  tiff = tiff.astype(np.float)
  tiff *= 255.
  tiff /= normalisation
  tiff = tiff.astype(np.uint8)
  return tiff
