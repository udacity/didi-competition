#!/bin/bash
set -e

# setup ros environment
source "/opt/ros/$ROS_DISTRO/setup.bash"
source "/workspace/install/setup.bash"
source /scripts/launch_viz.sh
exec "$@"
