# Tracklet Scripts

This folder contains scripts to create KITTI Tracklet files from rosbag capture data and evaluate Tracklet files.

## Installation

The scripts do not require installation. They can be run in-place or in Docker using included Dockerfile if you do not have a suitable ROS environment setup.

## Usage

If you are planning to use Docker, build the docker container manually or using ./build.sh

### bag_to_kitti.py -- Dump images and create KITTI Tracklet files

For running through Docker, you can use the helper script:
    ./run-bag_to_kitti.sh -i [local dir with folder containing bag file(s)] -o [local output dir] -- [args to pass to python script]

For example, if your dataset bags are in /data/bags/*.bag, and you'd like the output in /output:

    ./run-bag_to_kitti.sh -i /data/bags -o /output

The same as above, but you want to suppress image output:

    ./run-bag_to_kitti.sh -i /data/bags -o /output -- -m
    
Note that when passing paths via -i and -o through Docker THE PATHS MUST BE ABSOLUTE. You cannot map relative paths to Docker.

To run bag_to_kitti.py locally, the same -i -o arguments can be used and additional arguments listed in the help (-h) can also be used directly without passing via --. Any valid path, relative or absolute works when calling the script directly.
    
### evaluate_tracklets.py -- Evaluate predicted Tracklet against ground truth

The evaluate_tracklets script does not depend on a ROS environment so it's less relevant to run in the Docker environment.

Usage is straightforward. Run the script as per

    python evaluate_tracklets.py predicted_tracklet.xml ground_truth_tracklet.xml -o /output/metrics
    
If you don't want metrics output as csv files omit the -o argument.