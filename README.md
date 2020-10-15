# NanoSciTracker-Python

Nanoscience tracker prototyped in Python

## Dependencies

This project currently has the following dependences:

* OpenCV 4.4
* Scikit Learn
* Scikit Image
* Numpy
* Matplotlib

To install them using a conda environment:

```bash
# Create and set the environment
conda create mhpc
conda activate mhpc
# Install dependencies
conda install numpy matplotlib scikit-learn scikit-image shapely
conda install -c conda-forge opencv
```

## Running the project

### Local tracker

The local tracker accepts one of the quadrants of the mcherry video sequence.

To download a sample for analysis:

```bash
cd data
./download-data.sh
```

To run an analysis over the sample (assume the sample name is loaded in 
`SAMPLE`):

```bash
cd src/LocalTracker
./main.py --input ../../data/mcherry/$SAMPLE --draw_detection=1 --draw_tracking=1
```

### Global tracker

The global tracker creates a particle world with multi-scene capabilities.

To download a sample for analysis:

```bash
cd data
./download-data.sh
```

To run an analysis over the sample (assume the sample name is loaded in 
`SAMPLE`):

```bash
cd src/GlobalTracker
./main.py
```

The check the modifiers for `main.py`::

* `./main.py --help`

### ARES demo

This is the full demo of the project

To download a sample for analysis:

```bash
cd data
./download-data.sh
```

To run an analysis over the sample:

```bash
cd src/ares
# For single scene mode
./main.py --dataset=../data/mcherry/mcherry_single.json
# For multi-scene mode
./main.py --dataset=../data/mcherry/mcherry.json
```

The check the modifiers for `main.py`::

* `./main.py --help`

Version: 0.1.0

Author: Luis G. Leon-Vega
