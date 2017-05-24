# Point Cloud Visualization

This folder contains a docker environment and helper scripts for visualizing point clouds from bag files along with bounding box annotations in tracklet xml files in RVIZ. This environment currently relies on having a Linux host with nvidia-docker and an nvidia card and drivers installed. 

Contributions are welcome for hardware accelerated AMD/Intel solutions or working OpenGL software rending solutions.

Also, be aware, the rather insecure 'xhost+' command is utilized in the run.sh for giving unrestricted access to the host XWindows system for the Docker container. There are more secure methods but typically require matching host/container configuration steps to be performed.

## Installation

No installation required. Run ./build.sh to build the Docker container using defaults. The Velodyne drivers are built and installed in the container during this process.

## Usage

Run ./run.sh with -i argument set to a folder at the root of where both bag files and tracklets are located. All paths passed to the run.sh that are handed off to Docker MUST be absolute paths.

    ./run.sh -i /mydata/data 

Default behaviour will launch the ROS, launch the Velodyne node, launch RVIZ and leave you at the container's bash shell where you can run 'rosbag play' or run 'scripts/play.py <bag_file> <tracklet_file>' to play the bag file of your choice and see visualization in RVIZ. The play.py script allows visualization of the obstacles in a tracklet xml file.

It is possible to specify the bag to play from the host command line and skip using the shell in container as in this example:

    ./run.sh -i /mydata/data -r 'rosbag play /data/3.bag -l'

It is also possible to use scripts/play.py directly on your host OS if you have ROS and all other necessary Python modules installed.
    
   