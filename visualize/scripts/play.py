from __future__ import print_function
import numpy as np
import PyKDL as kd
import rospy
import rosbag
import argparse
import os
import sys
import subprocess
import colorsys

from collections import defaultdict
from parse_tracklet import *
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

# https://stackoverflow.com/questions/470690/how-to-automatically-generate-n-distinct-colors
kelly_colors_dict = dict(
    vivid_yellow=(255, 179, 0),
    strong_purple=(128, 62, 117),
    vivid_orange=(255, 104, 0),
    very_light_blue=(166, 189, 215),
    vivid_red=(193, 0, 32),
    grayish_yellow=(206, 162, 98),
    medium_gray=(129, 112, 102),
    vivid_green=(0, 125, 52),
    strong_purplish_pink=(246, 118, 142),
    strong_blue=(0, 83, 138),
    strong_yellowish_pink=(255, 122, 92),
    strong_violet=(83, 55, 122),
    vivid_orange_yellow=(255, 142, 0),
    strong_purplish_red=(179, 40, 81),
    vivid_greenish_yellow=(244, 200, 0),
    strong_reddish_brown=(127, 24, 13),
    vivid_yellowish_green=(147, 170, 0),
    deep_yellowish_brown=(89, 51, 21),
    vivid_reddish_orange=(241, 58, 19),
    dark_olive_green=(35, 44, 22))
kelly_colors_list = kelly_colors_dict.values()

CAMERA_TOPICS = ["/image_raw"]


class Frame():
    def __init__(self, trans, rotq, object_type, size):
        self.trans = trans
        self.rotq = rotq
        self.object_type = object_type
        self.size = size
        

class BoxSubPub():
    def __init__(self, frame_map, timestamp_map, sphere=False):
        self.frame_map = frame_map
        self.timestamp_map = timestamp_map
        rospy.Subscriber("/image_raw", Image, self._img_callback)
        self.publisher = rospy.Publisher("bbox", Marker)
        self.sphere = sphere

    def _publish_marker(
            self, 
            marker_type,
            marker_id,
            ts, 
            trans, 
            rotq,
            scale=[1.,1.,1.],
            color=[1.0,0.,0.]):
        mark = Marker()
        mark.header.frame_id = "velodyne"
        mark.header.stamp = ts
        mark.id = marker_id
        mark.type = marker_type
        mark.pose.position.x = trans[0]
        mark.pose.position.y = trans[1]
        mark.pose.position.z = trans[2]
        mark.pose.orientation.x = rotq[0]
        mark.pose.orientation.y = rotq[1]
        mark.pose.orientation.z = rotq[2]
        mark.pose.orientation.w = rotq[3]
        mark.scale.x = scale[0]
        mark.scale.y = scale[1]
        mark.scale.z = scale[2]
        mark.color.a = 0.5
        mark.color.r = color[0]
        mark.color.g = color[1]
        mark.color.b = color[2]
        self.publisher.publish(mark)

    def _img_callback(self, msg):
        frame_index = self.timestamp_map[msg.header.stamp.to_nsec()]
        for i, f in enumerate(self.frame_map[frame_index]):
            trans = f.trans
            rotq = f.rotq
            scale = f.size
            h, w, l = f.size
            color = kelly_colors_list[i % len(kelly_colors_list)]

            if self.sphere:
                s = max(l, w, h)
                self._publish_marker(
                    Marker.SPHERE, 0, msg.header.stamp,
                    trans, rotq, [s, s, s],
                    color)
                self._publish_marker(
                    Marker.ARROW, 1, msg.header.stamp,
                    trans, rotq, [l/2, w/2, h],
                    color)    
            else:
                #FIXME change Marker types based on object type (ie car vs pedestrian)
                self._publish_marker(
                    Marker.CUBE, 0, msg.header.stamp,
                    trans, rotq, [l, w, h],
                    color)
                self._publish_marker(
                    Marker.ARROW, 1, msg.header.stamp,
                    trans, rotq, [l/2, w/2, h],
                    color)


def extract_bag_timestamps(bag_file):
    timestamp_map = {}
    index = 0
    with rosbag.Bag(bag_file, "r") as bag:
        for topic, msg, ts in bag.read_messages(topics=CAMERA_TOPICS):
            timestamp_map[msg.header.stamp.to_nsec()] = index
            index += 1
    return timestamp_map


def generate_frame_map(tracklets):
        # map all tracklets to one timeline
    frame_map = defaultdict(list)
    for t in tracklets:
        for i in range(t.num_frames):
            frame_index = i + t.first_frame
            rot = t.rots[i]
            rotq = kd.Rotation.RPY(rot[0], rot[1], rot[2]).GetQuaternion()
            frame_map[frame_index].append(
                Frame(
                    t.trans[i],
                    rotq,
                    t.object_type,
                    t.size))
    return frame_map


def main():
    parser = argparse.ArgumentParser(description='Play bag and visualize tracklet.')
    parser.add_argument('bag', type=str, nargs='?', default='', help='bag filename')
    parser.add_argument('tracklet', type=str, nargs='?', default='tracklet_labels.xml', help='tracklet filename')
    parser.add_argument('-s', dest='sphere', action='store_true',
        help='Use sphere instead of default object_type bbox.')
    parser.set_defaults(sphere=False)
    args = parser.parse_args()
    sphere = args.sphere

    tracklet_file = args.tracklet
    if not os.path.exists(tracklet_file):
        sys.stderr.write('Error: Tracklet file %s not found.\n' % tracklet_file)
        exit(-1)
    tracklets = parse_xml(tracklet_file)
    if not tracklets:
        sys.stderr.write('Error: No Tracklets parsed.\n')
        exit(-1)

    bag_file = args.bag
    if not os.path.exists(bag_file):
        sys.stderr.write('Error: Bag file %s not found.\n' % bag_file)
        exit(-1)

    rospy.init_node("tracklet")  
    timestamp_map = extract_bag_timestamps(bag_file)
    frame_map = generate_frame_map(tracklets)
    BoxSubPub(frame_map, timestamp_map, sphere)
    subprocess.call(['rosbag','play', bag_file, '-l', '-q'])


if __name__ == '__main__':
    main()