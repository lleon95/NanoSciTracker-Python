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

'''
Development notes:
- The position should be the most relevant and it must be as close
  as possible
- The weights and threshold are hyperparameters
- The position is matched here since it's a global parameter
- The rest of features are local level parameters, they should be analysed
  within the trackers
- The analysis is from new to all out-of-scene

- The matcher will return the: not matched values in new and out and the
  new deployed trackers which matched
'''

import copy
import numpy as np

class Matcher:
    def __init__(self):
        # Features to analyse
        self.ce_position = True
        self.ce_velocity = True
        self.ce_angle = True
        self.ce_hog = False
        self.ce_histogram = True
        self.ce_mosse = False # FIXME
        
        # Weights
        self.w_position = 1
        self.w_velocity = 0.9
        self.w_angle = 1
        self.w_hog = 1.
        self.w_histogram = 1
        self.w_mosse = 1.
        
        # Probability Threshold
        self.threshold = 0.15
        self.max_death_time = 120
        
    def _compare_histogram(self, lhs, rhs):
        '''
        Computes the histogram probability lhs respect to rhs
        Params:
        - lhs, rhs: trackers
        Returns:
        - normalised probability (the closer, the higher)
        '''
        if not self.ce_histogram:
            return np.array([1.])
        
        histo1 = lhs.histogram
        histo2 = rhs.histogram
        
        # Perform comparison
        return histo2.compare(histo1)
    
    def _compare_mosse(self, lhs, rhs):
        '''
        Computes the mosse probability lhs respect to rhs
        Params:
        - lhs, rhs: trackers
        Returns:
        - normalised probability (the closer, the higher)
        '''
        if not self.ce_mosse:
            return np.array([1.])
        
        mosse1 = lhs.mosse
        mosse2 = rhs.mosse
        
        # Perform comparison
        return mosse2.compare(mosse1)
      
    def _compare_hog(self, lhs, rhs):
        '''
        Computes the hog probability lhs respect to rhs
        Params:
        - lhs, rhs: trackers
        Returns:
        - normalised probability (the closer, the higher)
        '''
        if not self.ce_hog:
            return np.array([1.])
        
        hog1 = lhs.hog
        hog2 = rhs.hog
        
        # Perform comparison
        return hog2.compare(hog1)
        
    def _compare_velocity(self, lhs, rhs):
        '''
        Computes the velocity probability lhs respect to rhs
        Params:
        - lhs, rhs: trackers
        Returns:
        - normalised probability (the closer, the higher)
        '''
        if not self.ce_velocity and not self.ce_angle:
            return np.array([1., 1.])
        
        vel1 = lhs.velocity
        vel2 = rhs.velocity
        
        # Set up the comparer
        vel2.compare_speed = self.ce_velocity
        vel2.compare_direction = self.ce_angle
        
        # Perform comparison
        features = vel2.compare(vel1)
        return features
    
    def _compare_position(self, lhs, rhs):
        '''
        Computes the position probability
        Params:
        - lhs, rhs: trackers
        Returns:
        - normalised probability (the closer, the higher)
        '''
        if not self.ce_position:
            return np.array([1.])

        if lhs.roi_offset is None:
            lroi = (0,0)
        else:
            lroi = lhs.roi_offset
        if rhs.roi_offset is None:
            rroi = (0,0)
        else:
            rroi = rhs.roi_offset

        lpos = [lhs.position[0] + lroi[0], lhs.position[1] + lroi[1]]
        rpos = [rhs.position[0] + rroi[0], rhs.position[1] + rroi[1]]

        X = np.array(lpos)
        Y = np.array(rpos)
        normaliser = np.linalg.norm([1200,1400])
        X = X/normaliser
        Y = Y/normaliser
        distance = np.linalg.norm(X - Y)
        
        return np.array([1 - distance])
    
    def clean(self, cur_v, new_v, out_v, last_idx, frame_cnt):
        for tracker in out_v:
            try:
                cur_v.remove(tracker)
            except:
                # The scene doesn't delete the out-of-scene immediately
                continue
            
        for out_ in out_v:
            out_.death_time += 1
            if out_.death_time == self.max_death_time:
                out_v.remove(out_)
                
        for new_ in new_v:
            # Labeling
            if new_.label is None:
                last_idx += 1
                new_.label = {"id": last_idx, "time": frame_cnt}
            # Adding to the current list
            if not new_ in cur_v:
                cur_v.append(new_)
                
        new_v = []
        return last_idx, cur_v, new_v, out_v
    
    def match(self, cur_v, new_v, out_v):
        '''
        Matcher
        Params:
        - the vectors
        Returns:
        - new_v: not matched
        - out_v: not matched
        - cur_v: updated list
        '''
        out_local = list(set(out_v))
        new_local = copy.copy(new_v)
        cur_local = copy.copy(cur_v)
        
        for new_ in new_v:
            n_old = len(out_local)
            if n_old == 0:
                break
            probabilities = np.zeros((n_old,), dtype=np.float32)
            cnt = 0
            
            # Find the probabilities of all the trackers
            for out_ in out_local:
                weights = np.zeros((6,), dtype=np.float32)
                # Compare position
                weights[0] = self.w_position * \
                    self._compare_position(new_, out_)[0]
                # Compare speed and direction
                velocity = self._compare_velocity(new_, out_)
                weights[1] = self.w_velocity * velocity[0]
                weights[2] = self.w_angle * velocity[1]
                # Compare hog
                weights[3] = self.w_hog * \
                    self._compare_hog(new_, out_)[0]
                # Compare histo
                weights[4] = self.w_histogram * \
                    self._compare_histogram(new_, out_)[0]
                # Compare mosse
                weights[5] = self.w_mosse * \
                    self._compare_mosse(new_, out_)[0]
                # Probability Superposition
                probabilities[cnt] = weights.prod()
                cnt += 1
            
            # Find the maximum (argmax)
            max_idx = np.argmax(probabilities)
            print("probs", probabilities)
            max_val = probabilities[max_idx]
            
            if max_val >= self.threshold:
                # Accept
                out_tracker = out_local[max_idx]
                new_tracker = new_
                new_tracker.label = out_tracker.label
                cur_local.append(new_tracker)
                # Remove from the lists
                new_local.remove(new_tracker)
                out_local.remove(out_tracker)
                
        return cur_local, new_local, out_local
    