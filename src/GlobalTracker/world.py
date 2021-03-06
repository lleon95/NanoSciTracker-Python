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
import Matcher.matcher as GlobalMatcher
import LocalTracker.drawutils as DrawUtils


class World:
    def __init__(self, settings=None):
        self._settings = settings
        self._scenes = []
        self._new_trackers = []
        self._current_trackers = []
        self._out_trackers = []
        self._dead_trackers = []
        self._last_id = 0
        self._frame_cnt = 0
        self._tracer = None

        if settings is None:
            raise RuntimeError("World settings are not valid")

    def spawn_scenes(self, rois, overlapping=0, sampling_rate=3):
        """
        Creates the scenes by setting the ROIS (extrinsic parameter for the
        external reference system)

        Params:
        * rois: list([[x0, x1],[y0, y1]])
        * overlapping: pixels of overlapping
        * sampling_rate: how many times the detector is deployed

        Return: None
        """
        for roi in rois:
            self._scenes.append(
                Scene.Scene(
                    ROI=roi, overlap=overlapping, detection_sampling=sampling_rate, 
                    settings=self._settings
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

    def _find_dead_trackers(self):
        '''
        This routine executes the matcher in order to look for dead trackers 
        in the world and link them in a similar fashion to the out-of-scene
        This perform operations on: current, new, and dead trackers
        '''

        weights = self._settings.set_if_defined("dead_tracker_weights", None)
        threshold = self._settings.set_if_defined("dead_tracker_threshold", None)
        death_time = self._settings.set_if_defined("dead_tracker_death_time", None)
        match_instance = GlobalMatcher.Matcher(weights, threshold, death_time)

        # Perform matching
        return match_instance.match(
            self._current_trackers, self._new_trackers, self._dead_trackers
        )

    def _update_current_trackers(self):
        """
        Updates the current trackers list, removing those trackers which
        requires to be analysed for matching
        """
        self._frame_cnt += 1

        weights = self._settings.set_if_defined("global_matcher_weights", None)
        threshold = self._settings.set_if_defined("global_matcher_threshold", None)
        death_time = self._settings.set_if_defined("global_matcher_death_time", None)
        match_instance = GlobalMatcher.Matcher(weights, threshold, death_time)

        # Perform cleaning of replicates - this avoids redundancies
        (
            self._current_trackers,
            self._new_trackers,
            self._out_trackers,
            self._dead_trackers,
        ) = match_instance.pre_clean(
            self._current_trackers,
            self._new_trackers,
            self._out_trackers,
            self._dead_trackers,
        )

        # Link dead trackers first
        res = self._find_dead_trackers()
        self._current_trackers, self._new_trackers, self._dead_trackers = res

        # Perform matching - out of scene
        res = match_instance.match(
            self._current_trackers, self._new_trackers, self._out_trackers
        )
        self._current_trackers, self._new_trackers, self._out_trackers = res

        # Perform post cleaning
        (
            self._last_id,
            self._current_trackers,
            self._new_trackers,
            self._out_trackers,
        ) = match_instance.post_clean(
            self._current_trackers,
            self._new_trackers,
            self._out_trackers,
            self._last_id,
            self._frame_cnt,
        )

    def update_trackers(self, frames=None):
        """
        Updates the scene and its trackers
        Params: frames (optional). If present, it also load the frames to the
        scenes
        Return: None
        """
        if not frames is None:
            self.load_frames(frames)

        self._dead_trackers = list([])
        for scene in self._scenes:
            cur, out, new, dead = scene.update()
            self._new_trackers += new
            self._out_trackers += out
            self._dead_trackers += dead

        self._update_current_trackers()

        # Add trace
        if not self._tracer is None:
            self._tracer.push(self._current_trackers, self._new_trackers, \
                self._out_trackers, self._dead_trackers)

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
        """
        Draw trackers on the world canvas
        This draws the world global tracker queues on the world canvas

        Params:
        * world: frame with the world

        Returns:
        * frame: copy of world with the overlay
        """

        frame = copy.deepcopy(world)
        frame = DrawUtils.draw_trackers(frame, self._current_trackers, (255, 255, 255))
        frame = DrawUtils.draw_trackers(frame, self._new_trackers, (255, 0, 0))
        frame = DrawUtils.draw_trackers(frame, self._out_trackers, (0, 0, 255))
        frame = DrawUtils.draw_trackers(frame, self._dead_trackers, (255, 0, 255))
        frame = DrawUtils.place_text(
            frame, "Cur: " + str(len(self._current_trackers)), (0, 30)
        )
        return frame

    def attach_tracer(self, tracer):
        self._tracer = tracer

    def dump_trackers(self):
        if not self._tracer is None:
            self._tracer.dump()
            
    def __del__(self):
        self.dump_trackers()
