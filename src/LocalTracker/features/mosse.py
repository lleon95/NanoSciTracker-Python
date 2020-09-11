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
import copy
import cv2 as cv

from features.feature import Feature

'''
Comparison predicates
'''

def PSR_Max(lhs, rhs, th=11.4):
    M = max(lhs, rhs)
    ret = M/th 
    if ret > 1:
        return 1
    else:
        return ret

def randWarp(win):
    '''
    Auxiliar function
    This generates a random warp transformation for the window image to get
    multiple perspectives and train the filter in a proper way, reducing the
    overfitting

    Input: a window to transform
    Output: window transformed randomly
    '''
    C = 0.1
    ang = np.random.uniform(-1. * C, C)
    s = np.sin(ang)
    c = np.cos(ang)
    
    warp_transform = \
    np.array([[c + np.random.uniform(-1. * C, C), -s + np.random.uniform(-1. * C, C), 0.], \
            [s + np.random.uniform(-1. * C, C), c + np.random.uniform(-1. * C, C), 0.]])
    
    h, w = win.shape
    center_warp = np.array([[w/2], [h/2]])
    warp_transform[:,2:] = center_warp - warp_transform[:,:2].dot(center_warp)
    
    return cv.warpAffine(win, M=warp_transform, dsize=(w, h), borderMode=cv.BORDER_REFLECT)

def preprocess(win, hanWin):
    '''
    Preprocess the image applying the hanningWindow on top of it

    Input: window to preprocess (win) and hanningWindow
    '''
    win.astype(np.float32)
    win = np.log(win + 1.)
    mean, std = cv.meanStdDev(win)
    win = (win - mean) / (std + .00001)
    
    return win * hanWin

class MosseFilter(Feature):
    def __init__(self, lr=0.2, th=5.7):
        super().__init__()

        # Filter details
        self.A = None
        self.B = None 
        self.H = None
        self.G = None
        self.hanWin = None
        self.size = None
        self.f = None
        self.PSR = 0
        self.last_frame = None
        self.center = None

        # Hyper-parameters
        self.lr = lr
        self.th = th

    def extractBoundingBox(self, bounding_box):
        p1, p2 = bounding_box
        w_bb = p2[0] - p1[0]
        h_bb = p2[1] - p1[1]

        h = cv.getOptimalDFTSize(h_bb)
        w = cv.getOptimalDFTSize(w_bb)

        return p1, w, h

    def initialise(self, gray_image, bounding_box):
        '''
        Receives a grayscale frame and a bounding box and operates over it
        return True if OK

        Bounding box format: ((x0, y0),(x1, y1))

        This uses the Numpy FFT since it already represents the numbers in
        complex representation and allows to do Spectrum Multiplciation and
        Vision in a straight-forward fashion
        '''
        self.last_frame = gray_image
        p1, w, h = self.extractBoundingBox(bounding_box)
        p1, p2 = bounding_box
        cols = p2[0] - p1[0]
        rows = p2[1] - p1[1]

        self.size = (w, h)
        x1 = int(round((2 * p1[0] + cols - w)/2))
        y1 = int(round((2 * p1[1] + rows - h)/2))
        self.center = (x1 + (w/2), y1 + (h/2))

        # Get inputs - window from BBox and the HanningWindow for gaussianity
        window = cv.getRectSubPix(gray_image, self.size, self.center)
        self.hanWin = cv.createHanningWindow(self.size, cv.CV_32F)

        # Create goal and its FFT
        g = np.zeros((h,w), np.float32)
        g[int(h/2)][int(w/2)] = 1.
        g = cv.GaussianBlur(g, (-1,-1), 2.0)
        maxVal = cv.minMaxLoc(g)[1]
        maxVal = 1. / maxVal
        g = g * maxVal
        self.G = np.fft.fft2(g)

        # Initialise the filter and other elements
        self.A = np.zeros(self.G.shape, dtype=np.complex128)
        self.B = np.zeros(self.G.shape, dtype=np.complex128)

        # Train the filter with a random warping
        for i in range(8):
            warped = randWarp(window)
            self.f = preprocess(warped, self.hanWin)
            F = np.fft.fft2(self.f)
            A_i = self.G * np.conjugate(F)
            B_i = F * np.conjugate(F)
            self.A += A_i
            self.B += B_i

        if np.isin(0 + 0j, self.B).any():
            return False
        
        self.H = self.A / self.B
        return True

    def predict(self, gray_image, bounding_box):
        '''
        Receives a grayscale frame and a bounding box and operates over it
        return True if the object matches with the template

        Bounding box format: ((x0, y0),(x1, y1))

        Warning: bounding box should be the same size

        Return True if matching and Bounding Box with location
        '''
        # Verify initialisation
        if self.H is None:
            return (False, bounding_box)
        
        w_f, h_f = self.size
        p = [0,0]

        # Extract the window
        if not bounding_box is None:
            p1, w, h = self.extractBoundingBox(bounding_box)
            
            p = [p1[0], p1[1]]

            c0 = p[0] + w/2
            c1 = p[1] + h/2
            self.center = (c0, c1)

            # Align bounding boxes if needed
            if w_f != w or h_f != h:
                p[0] = int(c[0] - (w_f/2))
                p[1] = int(c[1] - (h_f/2))

            self.last_frame = gray_image

        # Get window
        window = cv.getRectSubPix(self.last_frame, self.size, self.center)
        self.f = preprocess(window, self.hanWin)
        # Apply correlation
        F = np.fft.fft2(self.f)
        F_r = F * self.H
        f_r = np.real(np.fft.ifft2(F_r))
        # Find the PSR
        minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(f_r)
        delta_x = maxLoc[0] - w_f/2
        delta_y = maxLoc[1] - h_f/2
        mean = np.mean(f_r)
        std = np.std(f_r)
        PSR = (maxVal-mean) / (std + 0.00001)
        self.PSR = PSR

        # Threshold PSR
        if PSR < self.th:
            return (False, bounding_box)

        x0 = p[0] + delta_x
        y0 = p[1] + delta_y

        bbox = ((x0, y0),(x0 + w_f, y0 + h_f))
        return True, bbox

    def update(self, gray_image, bounding_box):
        '''
        Receives a grayscale frame and a bounding box and operates over it
        return True if the object matches with the template and updates the
        template

        Bounding box format: ((x0, y0),(x1, y1))

        Warning: bounding box should be the same size

        Return True if matching and Bounding Box with location
        '''
        # Verify initialisation
        if self.H is None:
            return False
        
        self.last_frame = gray_image

        # Make prediction
        res, bbox = self.predict(gray_image, bounding_box)

        if not res:
            return res

        # Update the filter
        p1, w, h = self.extractBoundingBox(bounding_box)
        self.center = (p1[0] + w/2, p1[1] + h/2)
        window_new = cv.getRectSubPix(gray_image, self.size, self.center)

        # Compute new F
        self.f = preprocess(window_new, self.hanWin)
        F = np.fft.fft2(self.f)

        # Extract the filter
        A_new = self.G * np.conjugate(F)
        B_new = F * np.conjugate(F)

        # Learn
        self.A = self.A*(1 - self.lr) + A_new * self.lr
        self.B = self.B*(1 - self.lr) + B_new * self.lr
        self.H = self.A / self.B

        return res

    def compare(self, mosse2, predicate=PSR_Max):
        '''
        To compare, it computes the similarity between the filters in order to
        see if they matches each other. The Threshold is set in times the
        defined threshold for the tracker
        '''
        # Back up the images
        swap = self.last_frame
        self.last_frame = mosse2.last_frame
        mosse2.last_frame = swap
        
        # Predict
        self.predict(None, None)
        mosse2.predict(None, None)
        
        # Get the PSRs
        PSR1 = self.PSR
        PSR2 = mosse2.PSR
        
        # Get the distribution. The addition should be greater than th_t
        ret = predicate(PSR1, PSR2)
        
        mosse2.last_frame = self.last_frame
        self.last_frame = swap
        
        return np.array([ret])        
        