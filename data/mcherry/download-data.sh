#!/bin/bash

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

echo "Downloading the data from the NFFA repository"

# Get the dataset
ls mcherry3.mp4 &> /dev/null || \
wget "https://datashare.nffa.eu/index.php/s/XS3Eb5tR6Ffmana/download" -O "mcherry3.mp4"
ls mcherry2.mp4 &> /dev/null || \
wget "https://datashare.nffa.eu/index.php/s/pXKNCiKQW6LfDyP/download" -O "mcherry2.mp4"
ls mcherry1.mp4 &> /dev/null || \
wget "https://datashare.nffa.eu/index.php/s/PY2RcALfYR3zAxj/download" -O "mcherry1.mp4"
ls mcherry0.mp4 &> /dev/null || \
wget "https://datashare.nffa.eu/index.php/s/PFqn4x83yPiJqtd/download" -O "mcherry0.mp4"
ls mcherry.avi &> /dev/null || \
wget "https://datashare.nffa.eu/index.php/s/DPNrr2bNS7BBGJJ/download" -O "mcherry.avi"

echo "Done!"
