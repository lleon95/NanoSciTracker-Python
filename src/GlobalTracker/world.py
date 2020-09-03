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

import copy

import scene as Scene
import LocalTracker.drawutils as DrawUtils

class World:
    def __init__(self):
        self._scenes = []
        self._new_trackers = []
        self._current_trackers = []
        self._out_trackers = []

    def spawn_scenes(self, rois, overlapping=0, sampling_rate=3):
        '''
        Creates the scenes by setting the ROIS (extrinsic parameter for the
        external reference system)

        Params:
        * rois: list([[x0, x1],[y0, y1]])
        * overlapping: pixels of overlapping
        * sampling_rate: how many times the detector is deployed

        Return: None
        '''
        for roi in rois:
            self._scenes.append(
                Scene.Scene(
                    ROI=roi, overlap=overlapping, detection_sampling=sampling_rate
                )
            )

    def load_frames(self, frames):
        """
        Load the frames to the scenes without performing any update
        Params: frames (array)
        Return: None
        """
        assert len(self._scenes) == len(frames)
        for i in range(len(self._scenes)):
            self._scenes[i].load_frame(frames[i])

    def update_trackers(self, frames=None):
        """
        Updates the scene and its trackers
        Params: frames (optional). If present, it also load the frames to the
        scenes
        Return: None
        """
        if not frames is None:
            self.load_frames(frames)

        for scene in self._scenes:
            cur, out, new = scene.update()
            self._new_trackers += new 
            self._out_trackers += out

        self._current_trackers += self._new_trackers

    def label_scenes(self):
        """
        Draw the tracking and detection within a copy of the frame
        Params: None
        Return: Frames in order

        Colour code:
        Purple: New detections
        Red: Detections
        Blue: Trackers
        Light blue: Out of scene
        """
        frames = []
        for scene in self._scenes:
            frame = copy.deepcopy(scene.frame)
            frames.append(scene.draw(frame))

        return frames

    def draw_trackers(self, world):
        frame = copy.deepcopy(world)
        return DrawUtils.draw_trackers(frame, self._current_trackers, \
            (255,255,255))
