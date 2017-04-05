#! /usr/bin/python
""" Udacity Self-Driving Car Challenge Bag Processing
"""

from __future__ import print_function
from cv_bridge import CvBridge, CvBridgeError
from collections import defaultdict
import os
import sys
import cv2
import imghdr
import argparse
import functools
import numpy as np
import pandas as pd
import PyKDL as kd

define_old_topics = True
from bag_topic_def import *
from bag_utils import *
from generate_tracklet import *


def get_outdir(base_dir, name=''):
    outdir = os.path.join(base_dir, name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def obs_prefix_from_topic(topic):
    words = topic.split('/')
    start, end = (1, 4) if topic.startswith(OBJECTS_TOPIC_ROOT) else (1, 3)
    prefix = '_'.join(words[start:end])
    name = words[2] if topic.startswith(OBJECTS_TOPIC_ROOT) else words[1]
    return prefix, name


def check_format(data):
    img_fmt = imghdr.what(None, h=data)
    return 'jpg' if img_fmt == 'jpeg' else img_fmt


def write_image(bridge, outdir, msg, fmt='png'):
    results = {}
    image_filename = os.path.join(outdir, str(msg.header.stamp.to_nsec()) + '.' + fmt)
    try:
        if hasattr(msg, 'format') and 'compressed' in msg.format:
            buf = np.ndarray(shape=(1, len(msg.data)), dtype=np.uint8, buffer=msg.data)
            cv_image = cv2.imdecode(buf, cv2.IMREAD_ANYCOLOR)
            if cv_image.shape[2] != 3:
                print("Invalid image %s" % image_filename)
                return results
            results['height'] = cv_image.shape[0]
            results['width'] = cv_image.shape[1]
            # Avoid re-encoding if we don't have to
            if check_format(msg.data) == fmt:
                buf.tofile(image_filename)
            else:
                cv2.imwrite(image_filename, cv_image)
        else:
            cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
            cv2.imwrite(image_filename, cv_image)
    except CvBridgeError as e:
        print(e)
    results['filename'] = image_filename
    return results


def camera2dict(msg, write_results, camera_dict):
    camera_dict["timestamp"].append(msg.header.stamp.to_nsec())
    if write_results:
        camera_dict["width"].append(write_results['width'] if 'width' in write_results else msg.width)
        camera_dict['height'].append(write_results['height'] if 'height' in write_results else msg.height)
        camera_dict["frame_id"].append(msg.header.frame_id)
        camera_dict["filename"].append(write_results['filename'])


def gps2dict(msg, gps_dict):
    gps_dict["timestamp"].append(msg.header.stamp.to_nsec())
    #gps_dict["status"].append(msg.status.status)
    #gps_dict["service"].append(msg.status.service)
    gps_dict["lat"].append(msg.latitude)
    gps_dict["long"].append(msg.longitude)
    gps_dict["alt"].append(msg.altitude)


def rtk2dict(msg, rtk_dict):
    rtk_dict["timestamp"].append(msg.header.stamp.to_nsec())
    rtk_dict["tx"].append(msg.pose.pose.position.x)
    rtk_dict["ty"].append(msg.pose.pose.position.y)
    rtk_dict["tz"].append(msg.pose.pose.position.z)
    rotq = kd.Rotation.Quaternion(
        msg.pose.pose.orientation.x,
        msg.pose.pose.orientation.y,
        msg.pose.pose.orientation.z,
        msg.pose.pose.orientation.w)
    rot_xyz = rotq.GetRPY()
    rtk_dict["rx"].append(0.0) #rot_xyz[0]
    rtk_dict["ry"].append(0.0) #rot_xyz[1]
    rtk_dict["rz"].append(rot_xyz[2])


def imu2dict(msg, imu_dict):
    imu_dict["timestamp"].append(msg.header.stamp.to_nsec())
    imu_dict["ax"].append(msg.linear_acceleration.x)
    imu_dict["ay"].append(msg.linear_acceleration.y)
    imu_dict["az"].append(msg.linear_acceleration.z)


def interpolate_to_camera(camera_df, other_dfs, filter_cols=[]):
    if not isinstance(other_dfs, list):
        other_dfs = [other_dfs]
    if not isinstance(camera_df.index, pd.DatetimeIndex):
        print('Error: Camera dataframe needs to be indexed by timestamp for interpolation')
        return pd.DataFrame()

    for o in other_dfs:
        o['timestamp'] = pd.to_datetime(o['timestamp'])
        o.set_index(['timestamp'], inplace=True)
        o.index.rename('index', inplace=True)

    merged = functools.reduce(lambda left, right: pd.merge(
        left, right, how='outer', left_index=True, right_index=True), [camera_df] + other_dfs)
    merged.interpolate(method='time', inplace=True, limit=100, limit_direction='both')

    filtered = merged.loc[camera_df.index]  # back to only camera rows
    filtered.fillna(0.0, inplace=True)
    filtered['timestamp'] = filtered.index.astype('int')  # add back original timestamp integer col
    if filter_cols:
        if not 'timestamp' in filter_cols:
            filter_cols += ['timestamp']
        filtered = filtered[filter_cols]

    return filtered


def estimate_obstacle_poses(
    cap_front_rtk,
    cap_front_gps_offset,
    cap_rear_rtk,
    cap_rear_gps_offset,
    obs_rear_rtk,
    obs_rear_gps_offset,  # offset along [l, w, h] dim of car
):
    # offsets are all [l, w, h] lists (or tuples)
    assert(len(cap_front_gps_offset) == 3)
    assert(len(cap_rear_gps_offset) == 3)
    assert(len(obs_rear_gps_offset) == 3)
    # all coordinate records should be interpolated to same sample base at this point
    assert len(cap_front_rtk) == len(cap_rear_rtk) == len(obs_rear_rtk)

    rtk_coords = zip(cap_front_rtk, cap_rear_rtk, obs_rear_rtk)
    output_poses = []
    for c in rtk_coords:
        # FIXME currently just passing obstacle pose through, this is not valid.
        # Capture vehicle front + rear coordinates and GPS offset values must be
        # used to calculate a position and orientation of the obstacle relative
        # to the body frame of the capture vehicle.
        output_poses.append(c[2])

    # Tracklet gen (consumer of these poses) is expecting list of dicts with tx,ty,tz,rx,ry,rz keys
    return output_poses


def main():
    parser = argparse.ArgumentParser(description='Convert rosbag to images and csv.')
    parser.add_argument('-o', '--outdir', type=str, nargs='?', default='/output',
        help='Output folder')
    parser.add_argument('-i', '--indir', type=str, nargs='?', default='/data',
        help='Input folder where bagfiles are located')
    parser.add_argument('-f', '--img_format', type=str, nargs='?', default='jpg',
        help='Image encode format, png or jpg')
    parser.add_argument('-m', dest='msg_only', action='store_true', help='Messages only, no images')
    parser.add_argument('-d', dest='debug', action='store_true', help='Debug print enable')
    parser.set_defaults(msg_only=False)
    parser.set_defaults(debug=False)
    args = parser.parse_args()

    img_format = args.img_format
    base_outdir = args.outdir
    indir = args.indir
    msg_only = args.msg_only
    debug_print = args.debug

    bridge = CvBridge()

    include_images = False if msg_only else True

    use_old_topics = False
    if use_old_topics:
        filter_topics = CAMERA_TOPICS + [OLD_RTK_TOPIC]
    else:
        filter_topics = CAMERA_TOPICS + [
            CAP_FRONT_GPS_TOPIC, CAP_REAR_GPS_TOPIC, CAP_FRONT_RTK_TOPIC, CAP_REAR_RTK_TOPIC]

    # For bag sets that may have missing metadata.csv file
    default_metadata = [{
        'obstacle_name': 'obs1',
        'object_type': 'Car',
        'gps_l': 2.032,
        'gps_w': 1.4478,
        'gps_h': 1.6256,
        'l': 4.2418,
        'w': 1.4478,
        'h': 1.5748,
    }]

    #FIXME scan from bag info in /obstacles/ topic path
    OBSTACLES = ['obs1']
    OBSTACLE_RTK_TOPICS = [OBJECTS_TOPIC_ROOT + '/' + x + '/rear/gps/rtkfix' for x in OBSTACLES]
    filter_topics += OBSTACLE_RTK_TOPICS

    bagsets = find_bagsets(indir, filter_topics=filter_topics, set_per_file=True, metadata_filename='metadata.csv')
    if not bagsets:
        print("No bags found in %s" % indir)
        exit(-1)

    for bs in bagsets:
        print("Processing set %s" % bs.name)
        sys.stdout.flush()

        camera_cols = ["timestamp", "width", "height", "frame_id", "filename"]
        camera_dict = defaultdict(list)

        gps_cols = ["timestamp", "lat", "long", "alt"]
        cap_rear_gps_dict = defaultdict(list)
        cap_front_gps_dict = defaultdict(list)

        rtk_cols = ["timestamp", "tx", "ty", "tz", "rx", "ry", "rz"]
        cap_rear_rtk_dict = defaultdict(list)
        cap_front_rtk_dict = defaultdict(list)

        # For the obstacles, keep track of rtk values for each one in a dictionary (key == topic)
        obstacle_rtk_dicts = {k: defaultdict(list) for k in OBSTACLE_RTK_TOPICS}

        dataset_outdir = os.path.join(base_outdir, "%s" % bs.name)
        get_outdir(dataset_outdir)
        print(dataset_outdir)
        if include_images:
            camera_outdir = get_outdir(dataset_outdir, "camera")
        bs.write_infos(dataset_outdir)
        readers = bs.get_readers()
        stats_acc = defaultdict(int)

        def _process_msg(topic, msg, stats):
            timestamp = msg.header.stamp.to_nsec()
            if topic in CAMERA_TOPICS:
                if debug_print:
                    print("%s_camera %d" % (topic[1], timestamp))

                write_results = {}
                if include_images:
                    write_results = write_image(bridge, camera_outdir, msg, fmt=img_format)
                    write_results['filename'] = os.path.relpath(write_results['filename'], dataset_outdir)
                camera2dict(msg, write_results, camera_dict)
                stats['img_count'] += 1
                stats['msg_count'] += 1

            elif topic == CAP_REAR_RTK_TOPIC:
                rtk2dict(msg, cap_rear_rtk_dict)
                stats['msg_count'] += 1

            elif topic == CAP_FRONT_RTK_TOPIC:
                rtk2dict(msg, cap_front_rtk_dict)
                stats['msg_count'] += 1

            elif topic == CAP_REAR_GPS_TOPIC:
                gps2dict(msg, cap_rear_gps_dict)
                stats['msg_count'] += 1

            elif topic == CAP_FRONT_GPS_TOPIC:
                gps2dict(msg, cap_front_gps_dict)
                stats['msg_count'] += 1

            elif topic in OBSTACLE_RTK_TOPICS:
                rtk2dict(msg, obstacle_rtk_dicts[topic])
                stats['msg_count'] += 1

            else:
                pass

        for reader in readers:
            for result in reader.read_messages():
                _process_msg(*result, stats=stats_acc)
                if ((stats_acc['img_count'] and stats_acc['img_count'] % 1000 == 0) or
                        (stats_acc['msg_count'] and stats_acc['msg_count'] % 10000 == 0)):
                    print("%d images, %d messages processed..." %
                          (stats_acc['img_count'], stats_acc['msg_count']))
                    sys.stdout.flush()

        print("Writing done. %d images, %d messages processed." %
              (stats_acc['img_count'], stats_acc['msg_count']))
        sys.stdout.flush()

        camera_df = pd.DataFrame(data=camera_dict, columns=camera_cols)
        if include_images:
            camera_df.to_csv(os.path.join(dataset_outdir, 'capture_vehicle_camera.csv'), index=False)

        cap_rear_gps_df = pd.DataFrame(data=cap_rear_gps_dict, columns=gps_cols).astype(float)
        cap_rear_gps_df.to_csv(os.path.join(dataset_outdir, 'capture_vehicle_rear_gps.csv'), index=False)

        cap_front_gps_df = pd.DataFrame(data=cap_front_gps_dict, columns=gps_cols).astype(float)
        cap_front_gps_df.to_csv(os.path.join(dataset_outdir, 'capture_vehicle_front_gps.csv'), index=False)

        cap_rear_rtk_df = pd.DataFrame(data=cap_rear_rtk_dict, columns=rtk_cols).astype(float)
        cap_rear_rtk_df.to_csv(os.path.join(dataset_outdir, 'capture_vehicle_rear_rtk.csv'), index=False)

        cap_front_rtk_df = pd.DataFrame(data=cap_front_rtk_dict, columns=rtk_cols).astype(float)
        cap_front_rtk_df.to_csv(os.path.join(dataset_outdir, 'capture_vehicle_front_rtk.csv'), index=False)

        obs_rtk_df_dict = {}
        for obs_topic, obs_rtk_dict in obstacle_rtk_dicts.items():
            obs_prefix, _ = obs_prefix_from_topic(obs_topic)
            obs_rtk_df = pd.DataFrame(data=obs_rtk_dict, columns=rtk_cols)
            obs_rtk_df.to_csv(os.path.join(dataset_outdir, '%s_rtk.csv' % obs_prefix), index=False)
            obs_rtk_df_dict[obs_topic] = obs_rtk_df

        if len(camera_dict['timestamp']):
            # Interpolate samples from all used sensors to camera frame timestamps
            camera_df['timestamp'] = pd.to_datetime(camera_df['timestamp'])
            camera_df.set_index(['timestamp'], inplace=True)
            camera_df.index.rename('index', inplace=True)

            camera_index_df = pd.DataFrame(index=camera_df.index).astype(float)

            cap_rear_gps_interp = interpolate_to_camera(camera_index_df, cap_rear_gps_df, filter_cols=gps_cols)
            cap_rear_gps_interp.to_csv(
                os.path.join(dataset_outdir, 'capture_vehicle_rear_gps_interp.csv'), header=True)

            cap_front_gps_interp = interpolate_to_camera(camera_index_df, cap_front_gps_df, filter_cols=gps_cols)
            cap_front_gps_interp.to_csv(
                os.path.join(dataset_outdir, 'capture_vehicle_front_gps_interp.csv'), header=True)

            cap_rear_rtk_interp = interpolate_to_camera(camera_index_df, cap_rear_rtk_df, filter_cols=rtk_cols)
            cap_rear_rtk_interp.to_csv(
                os.path.join(dataset_outdir, 'capture_vehicle_rear_rtk_interp.csv'), header=True)
            cap_rear_rtk_interp_rec = cap_rear_rtk_interp.to_dict(orient='records')

            cap_front_rtk_interp = interpolate_to_camera(camera_index_df, cap_front_rtk_df, filter_cols=rtk_cols)
            cap_front_rtk_interp.to_csv(
                os.path.join(dataset_outdir, 'capture_vehicle_front_rtk_interp.csv'), header=True)
            cap_front_rtk_interp_rec = cap_front_rtk_interp.to_dict(orient='records')

            collection = TrackletCollection()
            for obs_topic in obstacle_rtk_dicts.keys():
                obs_rtk_df = obs_rtk_df_dict[obs_topic]
                obs_interp = interpolate_to_camera(camera_index_df, obs_rtk_df, filter_cols=rtk_cols)
                obs_prefix, obs_name = obs_prefix_from_topic(obs_topic)
                obs_interp.to_csv(
                    os.path.join(dataset_outdir, '%s_rtk_interpolated.csv' % obs_prefix), header=True)

                # Extract lwh and object type from CSV metadata mapping file
                md = bs.metadata if bs.metadata else default_metadata
                for x in md:
                    if x['obstacle_name'] == obs_name:
                        mdr = x

                obs_tracklet = Tracklet(
                    object_type=mdr['object_type'], l=mdr['l'], w=mdr['w'], h=mdr['h'], first_frame=0)

                # Convert NED RTK coords of obstacle to capture vehicle body frame relative coordinates
                obs_tracklet.poses = estimate_obstacle_poses(
                    cap_front_rtk=cap_rear_rtk_interp_rec,
                    cap_front_gps_offset=[0.0, 0.0, 0.0],  # FIXME need this value
                    cap_rear_rtk=cap_front_rtk_interp_rec,
                    cap_rear_gps_offset=[0.0, 0.0, 0.0],  # FIXME need this value
                    obs_rear_rtk=obs_interp.to_dict(orient='records'),
                    obs_rear_gps_offset=[mdr['gps_l'], mdr['gps_w'], mdr['gps_h']],
                )

                collection.tracklets.append(obs_tracklet)

            tracklet_path = os.path.join(dataset_outdir, 'tracklet_labels.xml')
            collection.write_xml(tracklet_path)

if __name__ == '__main__':
    main()
