#!/bin/bash
roscore&
roslaunch --wait velodyne_pointcloud 32e_points.launch&
rosrun rviz rviz -f velodyne -d /scripts/default.rviz&
sleep 3
echo "Play appropriate bag file with 'rosbag play <filename> -l'"
