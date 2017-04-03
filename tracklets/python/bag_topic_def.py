""" ROS Bag file topic definitions
"""

SINGLE_CAMERA_TOPIC = "/image_raw"
CAMERA_TOPICS = [SINGLE_CAMERA_TOPIC]

OLD_GPS_TOPIC = "/gps/fix"
OLD_RTK_TOPIC = "/gps/rtkfix"

CAP_REAR_GPS_TOPIC = "/capture_vehicle/rear/gps/fix"
CAP_REAR_RTK_TOPIC = "/capture_vehicle/rear/gps/rtkfix"
CAP_FRONT_GPS_TOPIC = "/capture_vehicle/front/gps/fix"
CAP_FRONT_RTK_TOPIC = "/capture_vehicle/front/gps/rtkfix"

OBJECTS_TOPIC_ROOT = "/objects"
