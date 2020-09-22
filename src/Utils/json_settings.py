#!/usr/bin/env python3
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

import json


class Settings:
    def __init__(self, path):

        self.data = None

        with open(path) as f:
            try:
                self.data = json.load(f)
            except json.decoder.JSONDecodeError as err:
                print("JSON Error: ", err)

    def is_valid(self):
        return not self.data is None

    def set_if_defined(self, key, fallback):
        return self.data.get(key, fallback)


if __name__ == "__main__":
    path = "./example.json"
    settings = Settings(path)

    file_obj = settings.data

    if not file_obj is None:
        print("Contents:")
        print(file_obj)
