#!/bin/bash
roscore&
roslaunch --wait velodyne_pointcloud 32e_points.launch&
rosrun rviz rviz -f velodyne -d /scripts/default.rviz&
sleep 3
echo "Play bag file w/ tracklet using 'python scripts/play.py <bag_filename> <tracklet_filename>'"
