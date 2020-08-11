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

import numpy as np
import cv2 as cv
import copy
from scipy.ndimage.measurements import label

def binarise_otsu(img, b=1):
    '''
    Binarise image
    
    Parameters:
    * img: grayscale image
    
    Output:
    * binarised image
    '''
    # Get the shape (rows, cols)
    size = np.shape(img)
    size_batch = (int(size[0]/b), int(size[1]/b))
    
    for i in range(b):
        for j in range(b):
            y0 = size_batch[0] * j
            y1 = size_batch[0] * (j + 1)
            x0 = size_batch[1] * i
            x1 = size_batch[1] * (i + 1)
            crop = img[y0:y1,x0:x1]
            ret, thresh = cv.threshold(crop,0,255,cv.THRESH_OTSU)
            img[y0:y1,x0:x1] = thresh
    return img

def locate_maxima(img, k):
    '''
    Computes the local maxima through dilatation and opening
    
    Parameters:
    * img: binary image from binarisation
    * k: kernel size
    
    Returns:
    * opening: a binary image with local maxima
    
    '''
    kernel = np.ones((k,k),np.uint8)
    opening = cv.dilate(img,kernel,iterations=1)
    opening = cv.morphologyEx(opening,cv.MORPH_OPEN, kernel, iterations = 2)
    ret, opening = cv.threshold(opening,0,255,cv.THRESH_OTSU)
    return opening

def compute_k(size):
    '''
    Computes the kernel size for the maxima
    
    Standard playground size: 1280 × 960, k0 = 17 
    Minimum: 230 x 170
    '''
    h = size[0]
    w = size[1]
    
    # If the image gives a k < 3, just return 3
    if h <= 170 and w <= 230:
        return 3
    
    k1_1 = h * 17 / 960
    k1_2 = w * 17 / 1280
    
    k1 = int((k1_1 + k1_2) / 2)
    
    # Is it even?
    if (k1 % 2) == 0:
        k1 += 1
    
    return k1

def get_bbs(labels, padding=32, min_size=16):
    """
    Get the BBoxes list
    """
    bb_list = list([])
    for i in range(1, labels[1]+1):
        # Find pixels with each car label value
        nonzero = (labels[0] == i).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        if (np.max(nonzerox) - np.min(nonzerox)) >= min_size and (np.max(nonzerox) - np.min(nonzerox)) >= min_size:
            bbox = ((np.min(nonzerox)-padding, np.min(nonzeroy)-padding), (np.max(nonzerox)+padding, np.max(nonzeroy)+padding))
            bb_list.append((bbox[0], bbox[1]))
    return bb_list

def compute_padding(size):
    '''
    Computes the padding size for the bboxes
    
    Standard playground size: 1280 × 960, k0 = 35
    '''
    h = size[0]
    w = size[1]
    
    p1_1 = h * 48 / 960
    p1_2 = w * 48 / 1280
    
    p1 = int((p1_1 + p1_2) / 2)
    
    return p1

def label_boxes(markers, size=None):
    heat = np.zeros_like(markers[:,:]).astype(np.float) 
    heat[markers == 1] = 255

    labels = label(heat)
    
    if size is None:
        size = np.shape(heat)

    padding = compute_padding(size)
    bb_list = get_bbs(labels, padding, padding)
    
    return bb_list

def bounding_boxes(negative, size=None):
    negative[negative == 255] = 1
    bb_list = label_boxes(negative, size)
    return bb_list

def detect(img, batches=2, size=None):
    '''
    Performs the detection by using binarisation and thresholding. It's
    principle is based on Otsu's thresholding followed by local maxima
    detection and thresholding again to make it binary.
    
    Parameters:
    
    img: grayscale image
    
    Return:
    
    bboxes
    '''
    # Binarise image with otsu with 4 windows (2^2)
    otsu = binarise_otsu(img, batches)

    if size is None:
        size = np.shape(otsu)

    # Locate the maxima
    k = compute_k(np.shape(otsu))
    maxima = locate_maxima(otsu, k)
    # Get the bounding boxes
    return bounding_boxes(maxima, size)