<img src="images/urdf.png" alt="MKZ Model" width="800px">

The repository holds the data required for getting started with the Udacity/Didi self-driving car challenge. To generate tracklets (annotation data) from the released datasets, check out the Docker code in the ```/tracklet/``` folder. For sensor transform information, check out ```mkz-description```. 

Please note that tracklets cannot be generated without modifying this code, as we added an additional RTK GPS receiver onto the capture vehicle in order to determine orientation.

## Resources
Here's a list of the projects we've open sourced already that may be helpful:
* [**ROS Examples**](https://github.com/mjshiggins/ros-examples) – Example ROS nodes for consuming/processing the released datasets (work in progress)
* [**Annotated Driving Datasets**](https://github.com/udacity/self-driving-car/tree/master/annotations) – Many hours of labelled driving data
* [**Driving Datasets**](https://github.com/udacity/self-driving-car/tree/master/datasets) – Over 10 hours of driving data (LIDAR, camera frames and more)
