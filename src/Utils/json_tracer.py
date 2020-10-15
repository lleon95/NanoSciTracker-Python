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
import json

'''
The JSON object is composed by the following structure:

[ <- frame array
  [ <- tracker array
    {
      "label": Number,
      "abs_position": [x, y],
      "rel_position": [x, y],
      "speed": [dx, dy],
      "direction": Number,
      "col_histogram": [Number, Number, ..., Number],
      "hog_histogram": [Number, Number, ..., Number],
      "status": Number,
      "spawn_time": Number
    }
  ]
]

where "status":
  0. Current
  1. New
  2. Out
  3. Dead
'''

class Tracer:
    def __init__(self, settings):
        self.__data = []
        self.__tracers = settings.set_if_defined("enable_tracer", [])
        self.__status = settings.set_if_defined("trace_status", [])
        self.__prefix = settings.set_if_defined("file_prefix", "results")

    def reset(self):
        self.__data = []

    def _compute_abs_position(self, tracker):
        if tracker.roi_offset is None:
            roi = (0, 0)
        else:
            roi = tracker.roi_offset

        return tracker.position[0] + roi[0], tracker.position[1] + roi[1]

    def _compute_speed(self, tracker):
        dx = round(tracker.velocity.speed[0].speed, 3)
        dy = round(tracker.velocity.speed[1].speed, 3)

        return dx, dy

    def _create_entry(self, tracker, status):
        entry = {}
        entry["status"] = status
        if not tracker.label is None:
            entry["label"] = tracker.label["id"]
            entry["spawn_time"] = tracker.label["time"]
        else:
            entry["label"] = -1
            entry["spawn_time"] = len(self.__data)

        if "rel_position" in self.__tracers:
            entry["rel_position"] = copy.deepcopy(tracker.position)
        if "abs_position" in self.__tracers:
            entry["abs_position"] = self._compute_abs_position(tracker)
        if "speed" in self.__tracers:
            entry["speed"] = copy.deepcopy(self._compute_speed(tracker))
        if "direction" in self.__tracers:
            entry["direction"] = round(tracker.velocity.direction, 3)
        if "col_histogram" in self.__tracers:
            entry["col_histogram"] = copy.deepcopy(tracker.histogram.histogram).tolist()
        if "hog_histogram" in self.__tracers:
            entry["hog_histogram"] = copy.deepcopy(tracker.hog.hog).tolist()
        return entry

    def push(self, current, new, out, dead):
        frame = []

        # Add current trackers to the frame
        if 0 in self.__status:
            for tracker in current:
                frame.append(self._create_entry(tracker, 0))
        # Add new trackers to the frame
        if 1 in self.__status:
            for tracker in new:
                frame.append(self._create_entry(tracker, 1))
        # Add out trackers to the frame
        if 2 in self.__status:
            for tracker in out:
                frame.append(self._create_entry(tracker, 2))
        # Add dead trackers to the frame
        if 3 in self.__status:
            for tracker in dead:
                frame.append(self._create_entry(tracker, 3))

        self.__data.append(frame)

    def dump(self):
        file_name = self.__prefix
        file_name += ".json"
        with open(file_name, "w") as outfile: 
            json.dump(self.__data, outfile)
