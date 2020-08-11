# NanoSciTracker-Python

Nano science tracker prototyped in Python

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
conda install numpy matplotlib scikit-learn scikit-image
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
./main.py --input ../../data/$SAMPLE --draw_detection=1 --draw_tracking=1
```

The modifiers for `main.py` are:

* `--input`: loads the video sequence
* `--draw_detection`: shows the detection on a window
* `--draw_tracking`: shows the tracking on a window
* `--sample_tracking`: sets the frame number to set deploy the tracking

Version: 0.1.0
Author: Luis G. Leon-Vega
