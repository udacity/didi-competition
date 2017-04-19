#! /usr/bin/python
""" Udacity Self-Driving Car Challenge Bag Processing
"""

from __future__ import print_function
from collections import defaultdict
import os
import sys
import math
import argparse
import functools
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PyKDL as kd

from bag_topic_def import *
from bag_utils import *
from generate_tracklet import *
from scipy.spatial import kdtree
from scipy import stats


# Bag message timestamp source
TS_SRC_PUB = 0
TS_SRC_REC = 1
TS_SRC_OBS_REC = 2

# Correction method
CORRECT_NONE = 0
CORRECT_PLANE = 1

CAP_RTK_FRONT_Z = .3323 + 1.2192
CAP_RTK_REAR_Z = .3323 + .8636


def get_outdir(base_dir, name=''):
    outdir = os.path.join(base_dir, name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def obs_name_from_topic(topic):
    return topic.split('/')[2]


def obs_prefix_from_topic(topic):
    words = topic.split('/')
    prefix = '_'.join(words[1:4])
    name = words[2]
    return prefix, name


def camera2dict(timestamp, msg, write_results, camera_dict):
    camera_dict["timestamp"].append(timestamp)
    if write_results:
        camera_dict["width"].append(write_results['width'] if 'width' in write_results else msg.width)
        camera_dict['height'].append(write_results['height'] if 'height' in write_results else msg.height)
        camera_dict["frame_id"].append(msg.header.frame_id)
        camera_dict["filename"].append(write_results['filename'])


def gps2dict(timestamp, msg, gps_dict):
    gps_dict["timestamp"].append(timestamp)
    gps_dict["lat"].append(msg.latitude)
    gps_dict["long"].append(msg.longitude)
    gps_dict["alt"].append(msg.altitude)


def rtk2dict(timestamp, msg, rtk_dict):
    rtk_dict["timestamp"].append(timestamp)
    rtk_dict["tx"].append(msg.pose.pose.position.x)
    rtk_dict["ty"].append(msg.pose.pose.position.y)
    rtk_dict["tz"].append(msg.pose.pose.position.z)
    rotq = kd.Rotation.Quaternion(
        msg.pose.pose.orientation.x,
        msg.pose.pose.orientation.y,
        msg.pose.pose.orientation.z,
        msg.pose.pose.orientation.w)
    rot_xyz = rotq.GetRPY()
    rtk_dict["rx"].append(0.0)
    rtk_dict["ry"].append(0.0)
    rtk_dict["rz"].append(rot_xyz[2])


def imu2dict(timestamp, msg, imu_dict):
    imu_dict["timestamp"].append(timestamp)
    imu_dict["ax"].append(msg.linear_acceleration.x)
    imu_dict["ay"].append(msg.linear_acceleration.y)
    imu_dict["az"].append(msg.linear_acceleration.z)


def get_yaw(p1, p2):
    return math.atan2(p1[1] - p2[1], p1[0] - p2[0])


def dict_to_vect(di):
    return kd.Vector(di['tx'], di['ty'], di['tz'])


def list_to_vect(li):
    return kd.Vector(li[0], li[1], li[2])


def frame_to_dict(frame):
    r, p, y = frame.M.GetRPY()
    return dict(tx=frame.p[0], ty=frame.p[1], tz=frame.p[2], rx=r, ry=p, rz=y)


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


def get_obstacle_pose(
        cap_front,
        cap_rear,
        obs_front,
        obs_rear,
        obs_gps_to_centroid,
        velodyne_to_front,
        cap_yaw_error_rad=0,
        cap_pitch_error_rad=0):

    # calculate capture yaw in ENU frame and setup correction rotation
    cap_front_v = dict_to_vect(cap_front)
    cap_rear_v = dict_to_vect(cap_rear)
    cap_yaw = get_yaw(cap_front_v, cap_rear_v)
    cap_yaw += cap_yaw_error_rad
    rot_cap = kd.Rotation.EulerZYX(-cap_yaw, -cap_pitch_error_rad, 0)

    obs_rear_v = dict_to_vect(obs_rear)
    if obs_front:
        obs_front_v = dict_to_vect(obs_front)
        obs_yaw = get_yaw(obs_front_v, obs_rear_v)
        # use the front gps as the obstacle reference point if it exists as it's closers
        # to the centroid and mounting metadata seems more reliable
        cap_to_obs = obs_front_v - cap_front_v
    else:
        cap_to_obs = obs_rear_v - cap_front_v

    # transform capture car to obstacle vector into capture car velodyne lidar frame
    res = rot_cap * cap_to_obs
    res += list_to_vect(velodyne_to_front)

    # obs_gps_to_centroid is offset for front gps if it exists, otherwise rear
    obs_gps_to_centroid_v = list_to_vect(obs_gps_to_centroid)
    if obs_front:
        # if we have both front + rear RTK calculate an obstacle yaw and use it for centroid offset
        obs_rot_z = kd.Rotation.RotZ(obs_yaw - cap_yaw)
        centroid_offset = obs_rot_z * obs_gps_to_centroid_v
    else:
        # if no obstacle yaw calculation possible, treat rear RTK as centroid and offset in Z only
        obs_rot_z = kd.Rotation()
        centroid_offset = kd.Vector(0, 0, obs_gps_to_centroid_v[2])
    res += centroid_offset

    return frame_to_dict(kd.Frame(obs_rot_z, res))


def check_oneof_topics_present(topic_map, name, topics):
    if not isinstance(topics, list):
        topics = [topics]
    if not any(t in topic_map for t in topics):
        print('Error: One of %s must exist in bag, skipping bag %s.' % (topics, name))
        return False
    return True


def filter_outliers(points):
    kt = kdtree.KDTree(points)
    distances, i = kt.query(kt.data, k=9)
    z_distances = stats.zscore(np.mean(distances, axis=1))
    o_filter = abs(z_distances) < 1  # rather arbitrary
    return points[o_filter]


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    http://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector
    """
    axis = np.asarray(axis)
    axis /= math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


def fit_plane(points, do_plot=True, dataset_outdir='', name='', debug=True):
    if debug:
        print('Processing %d points' % points.shape[0])

    centroid = np.mean(points, axis=0)
    if debug:
        print('centroid', centroid)
    points -= centroid

    _, _, v = np.linalg.svd(points)
    line = v[0] / np.linalg.norm(v[0])
    norm = v[-1] / np.linalg.norm(v[-1])
    norm *= np.sign(v[-1][-1])
    use_line = False
    if np.argmax(norm) != 2:
        print('Warning: Z component of plane normal is not largest, plane fit likely not optimal. Fitting line instead.')
        use_line = True
    if debug:
        print('line', line)
        print('norm', norm)

    if use_line:
        # find a rotation axis perpendicular to the fit line of the coords and
        # calculate rotation angle around that axis that levels fit line in z
        axis = np.cross(line, np.array([0, 0, 1.]))
        angle = line[2]
    else:
        # use plane normal to calculate a rotation axis and angle necessary
        # to level points in z
        z_cross_norm = np.cross(np.array([0, 0, 1.]), norm)
        angle = np.arcsin(np.linalg.norm(z_cross_norm))
        axis = z_cross_norm / np.linalg.norm(z_cross_norm)
    if debug:
        print('rotation', angle, axis)

    rot_m = rotation_matrix(axis, -angle)

    if do_plot:
        x_max, x_min = max(points[:, 0]), min(points[:, 0])
        y_max, y_min = max(points[:, 1]), min(points[:, 1])
        xy_max = max(x_max, y_max)
        xy_min = min(x_min, y_min)

        # compute normal of corrected points to visualize and verify
        points_rot = np.dot(rot_m, points.T).T
        _, _, vr = np.linalg.svd(points_rot)
        norm_rot = vr[-1] / np.linalg.norm(vr[-1])

        # build plane surface for original points and best fit plane for plotting
        # NOTE if line fit was used instead, this still just plots the plane fit
        d = np.array([0, 0, 0]).dot(norm)
        dr = np.array([0, 0, 0]).dot(norm_rot)
        xg, yg = np.meshgrid(
            range(int(x_min*1.3), int(math.ceil(x_max*1.3))),
            range(int(y_min*1.3), int(math.ceil(y_max*1.3))))
        zg = (d - norm[0] * xg - norm[1] * yg) * 1. / norm[2]
        zgr = (dr - norm_rot[0] * xg - norm_rot[1] * yg) * 1. / norm_rot[2]

        line_pts = line * np.mgrid[xy_min:xy_max:2j][:, np.newaxis]
        axis_pts = axis * np.mgrid[-xy_min:xy_min:2j][:, np.newaxis]
        norm_pts = norm * np.mgrid[-xy_min:xy_min:2j][:, np.newaxis]

        if False:
            points += centroid
            points_rot += centroid
            line_pts += centroid
            axis_pts += centroid
            norm_pts += centroid
            xg += int(centroid[0])
            yg += int(centroid[1])
            zg += centroid[2]
            zgr += centroid[2]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(*points.T)
        ax.scatter(*points_rot.T, c='y')
        ax.plot3D(*line_pts.T, c='r')
        ax.plot3D(*axis_pts.T, c='c')
        #ax.plot3D(*norm_pts.T, c='g')
        ax.plot_surface(xg, yg, zg, alpha=0.3)
        ax.plot_surface(xg, yg, zgr, alpha=0.3, color='y')
        angles = [0, 30, 60]
        elev = [0, 15, 30]
        for a in angles:
            for e in elev:
                ax.view_init(elev=e, azim=a)
                fig.savefig(os.path.join(dataset_outdir, '%s-%d-%d-plot.png' % (name, e, a)))
        plt.close(fig)

    return centroid, norm, rot_m


def extract_metadata(md, obs_name):
    md = next(x for x in md if x['obstacle_name'] == obs_name)
    if 'gps_l' in md:
        # make old rear RTK only obstacle metadata compatible with new
        md['rear_gps_l'] = md['gps_l']
        md['rear_gps_w'] = md['gps_w']
        md['rear_gps_h'] = md['gps_h']
    return md


def main():
    parser = argparse.ArgumentParser(description='Convert rosbag to images and csv.')
    parser.add_argument('-o', '--outdir', type=str, nargs='?', default='/output',
        help='Output folder')
    parser.add_argument('-i', '--indir', type=str, nargs='?', default='/data',
        help='Input folder where bagfiles are located')
    parser.add_argument('-f', '--img_format', type=str, nargs='?', default='jpg',
        help='Image encode format, png or jpg')
    parser.add_argument('-t', '--ts_src', type=str, nargs='?', default='pub',
        help="""Timestamp source. 'pub'=capture node publish time, 'rec'=receiver bag record time,
        'obs_rec'=record time for obstacles topics only, pub for others. Default='pub'""")
    parser.add_argument('-c', '--correct', type=str, nargs='?', default='',
        help="""Correction method. ''=no correction, 'plane'=fit plane to RTK coords and level. Default=''""")
    parser.add_argument('--yaw_err', type=float, nargs='?', default='0.0',
        help="""Amount in degrees to compensate for RTK based yaw measurement. Default=0.0'""")
    parser.add_argument('--pitch_err', type=float, nargs='?', default='0.0',
        help="""Amount in degrees to compensate for RTK based yaw measurement. Default=0.0.""")
    parser.add_argument('-m', dest='msg_only', action='store_true', help='Messages only, no images')
    parser.set_defaults(msg_only=False)
    parser.set_defaults(debug=False)
    args = parser.parse_args()

    img_format = args.img_format
    base_outdir = args.outdir
    indir = args.indir
    ts_src = TS_SRC_PUB
    if args.ts_src == 'rec':
        ts_src = TS_SRC_REC
    elif args.ts_src == 'obs_rec':
        ts_src = TS_SRC_OBS_REC
    correct = CORRECT_NONE
    if args.correct == 'plane':
        correct = CORRECT_PLANE
    yaw_err = args.yaw_err
    pitch_err = args.pitch_err
    msg_only = args.msg_only
    image_bridge = ImageBridge()

    include_images = False if msg_only else True

    filter_topics = CAMERA_TOPICS + CAP_FRONT_RTK_TOPICS + CAP_REAR_RTK_TOPICS \
        + CAP_FRONT_GPS_TOPICS + CAP_REAR_GPS_TOPICS

    # FIXME scan from bag info in /obstacles/ topic path
    OBSTACLES = ['obs1']
    OBS_FRONT_RTK_TOPICS = [OBJECTS_TOPIC_ROOT + '/' + x + '/front/gps/rtkfix' for x in OBSTACLES]
    OBS_REAR_RTK_TOPICS = [OBJECTS_TOPIC_ROOT + '/' + x + '/rear/gps/rtkfix' for x in OBSTACLES]
    filter_topics += OBS_FRONT_RTK_TOPICS
    filter_topics += OBS_REAR_RTK_TOPICS

    bagsets = find_bagsets(indir, filter_topics=filter_topics, set_per_file=True, metadata_filename='metadata.csv')
    if not bagsets:
        print("No bags found in %s" % indir)
        exit(-1)

    for bs in bagsets:
        print("Processing set %s" % bs.name)
        sys.stdout.flush()

        if not check_oneof_topics_present(bs.topic_map, bs.name, CAP_FRONT_RTK_TOPICS):
            continue
        if not check_oneof_topics_present(bs.topic_map, bs.name, CAP_REAR_RTK_TOPICS):
            continue

        camera_cols = ["timestamp", "width", "height", "frame_id", "filename"]
        camera_dict = defaultdict(list)

        gps_cols = ["timestamp", "lat", "long", "alt"]
        cap_rear_gps_dict = defaultdict(list)
        cap_front_gps_dict = defaultdict(list)

        rtk_cols = ["timestamp", "tx", "ty", "tz", "rx", "ry", "rz"]
        cap_rear_rtk_dict = defaultdict(list)
        cap_front_rtk_dict = defaultdict(list)

        # For the obstacles, keep track of rtk values for each one in a dictionary (key == topic)
        obstacle_rtk_dicts = {k: {'front': defaultdict(list), 'rear': defaultdict(list)} for k in OBSTACLES}

        dataset_outdir = os.path.join(base_outdir, "%s" % bs.name)
        get_outdir(dataset_outdir)
        if include_images:
            camera_outdir = get_outdir(dataset_outdir, "camera")
        bs.write_infos(dataset_outdir)
        readers = bs.get_readers()
        stats_acc = defaultdict(int)

        def _process_msg(topic, msg, ts_recorded, stats):
            timestamp = msg.header.stamp.to_nsec()  # default to publish timestamp in message header
            if ts_src == TS_SRC_REC:
                timestamp = ts_recorded.to_nsec()
            elif ts_src == TS_SRC_OBS_REC and topic in OBS_REAR_RTK_TOPICS:
                timestamp = ts_recorded.to_nsec()

            if topic in CAMERA_TOPICS:
                write_results = {}
                if include_images:
                    write_results = image_bridge.write_image(camera_outdir, msg, fmt=img_format)
                    write_results['filename'] = os.path.relpath(write_results['filename'], dataset_outdir)
                camera2dict(timestamp, msg, write_results, camera_dict)
                stats['img_count'] += 1
                stats['msg_count'] += 1

            elif topic in CAP_REAR_RTK_TOPICS:
                rtk2dict(timestamp, msg, cap_rear_rtk_dict)
                stats['msg_count'] += 1

            elif topic in CAP_FRONT_RTK_TOPICS:
                rtk2dict(timestamp, msg, cap_front_rtk_dict)
                stats['msg_count'] += 1

            elif topic in CAP_REAR_GPS_TOPICS:
                gps2dict(timestamp, msg, cap_rear_gps_dict)
                stats['msg_count'] += 1

            elif topic in CAP_FRONT_GPS_TOPICS:
                gps2dict(timestamp, msg, cap_front_gps_dict)
                stats['msg_count'] += 1

            elif topic in OBS_REAR_RTK_TOPICS:
                name = obs_name_from_topic(topic)
                rtk2dict(timestamp, msg, obstacle_rtk_dicts[name]['rear'])
                stats['msg_count'] += 1

            elif topic in OBS_FRONT_RTK_TOPICS:
                name = obs_name_from_topic(topic)
                rtk2dict(timestamp, msg, obstacle_rtk_dicts[name]['front'])
                stats['msg_count'] += 1

            else:
                pass

        for reader in readers:
            last_img_log = 0
            last_msg_log = 0
            for result in reader.read_messages():
                _process_msg(*result, stats=stats_acc)
                if last_img_log != stats_acc['img_count'] and stats_acc['img_count'] % 1000 == 0:
                    print("%d images, processed..." % stats_acc['img_count'])
                    last_img_log = stats_acc['img_count']
                    sys.stdout.flush()
                if last_msg_log != stats_acc['msg_count'] and stats_acc['msg_count'] % 10000 == 0:
                    print("%d messages processed..." % stats_acc['msg_count'])
                    last_msg_log = stats_acc['msg_count']
                    sys.stdout.flush()

        print("Writing done. %d images, %d messages processed." %
              (stats_acc['img_count'], stats_acc['msg_count']))
        sys.stdout.flush()

        camera_df = pd.DataFrame(data=camera_dict, columns=camera_cols)
        if include_images:
            camera_df.to_csv(os.path.join(dataset_outdir, 'capture_vehicle_camera.csv'), index=False)

        def init_and_save(data_dict, cols, filename):
            df = pd.DataFrame(data=data_dict, columns=cols)
            if len(df.index):
                df.to_csv(os.path.join(dataset_outdir, filename), index=False)
            return df

        cap_rear_gps_df = init_and_save(cap_rear_gps_dict, gps_cols, 'capture_vehicle_rear_gps.csv')
        cap_front_gps_df = init_and_save(cap_front_gps_dict, gps_cols, 'capture_vehicle_front_gps.csv')
        cap_rear_rtk_df =init_and_save(cap_rear_rtk_dict, rtk_cols, 'capture_vehicle_rear_rtk.csv')
        cap_front_rtk_df = init_and_save(cap_front_rtk_dict,rtk_cols, 'capture_vehicle_front_rtk.csv')
        if not len(cap_rear_rtk_df.index):
            print('Error: No capture vehicle rear RTK entries exist.'
                  'Skipping bag %s.' % bag.name)
            continue
        if not len(cap_rear_rtk_df.index):
            print('Error: No capture vehicle front RTK entries exist.'
                  'Skipping bag %s.' % bag.name)
            continue

        rtk_z_offsets = [np.array([0., 0., CAP_RTK_FRONT_Z]), np.array([0., 0., CAP_RTK_REAR_Z])]
        if correct > 0:
            # Correction algorithm attempts to fit plane to rtk measurements across both capture rtk
            # units and all obstacles. We will subtract known RTK unit mounting heights first.
            cap_front_points = cap_front_rtk_df.as_matrix(columns=['tx', 'ty', 'tz']) - rtk_z_offsets[0]
            cap_rear_points = cap_rear_rtk_df.as_matrix(columns=['tx', 'ty', 'tz']) - rtk_z_offsets[1]
            point_arrays = [cap_front_points, cap_rear_points]
            filtered_point_arrays = [filter_outliers(cap_front_points), filter_outliers(cap_rear_points)]

        obs_rtk_df_dict = {}
        for obs_name, obs_rtk_dict in obstacle_rtk_dicts.items():
            obs_rear_rtk_df = init_and_save(obs_rtk_dict['rear'], rtk_cols, '%s_rear_rtk.csv' % obs_name)
            obs_front_rtk_df = init_and_save(obs_rtk_dict['front'], rtk_cols, '%s_front_rtk.csv' % obs_name)
            if not len(obs_rear_rtk_df.index):
                print('Warning: No entries for obstacle %s in %s. Skipping.' % (obs_name, bs.name))
                continue
            obs_rtk_df_dict[obs_name] = {'rear': obs_rear_rtk_df}
            if len(obs_front_rtk_df.index):
                obs_rtk_df_dict[obs_name]['front'] = obs_front_rtk_df
            if correct > 0:
                # Use obstacle metadata to determine rtk mounting height and subtract that height
                # from obstacle readings
                md = extract_metadata(bs.metadata, obs_name)
                if not md:
                    print('Error: No metadata found for %s, skipping obstacle.' % obs_name)
                    continue
                obs_z_offset = np.array([0., 0., md['rear_gps_h']])
                rtk_z_offsets.append(obs_z_offset)
                obs_rear_points = obs_rear_rtk_df.as_matrix(columns=['tx', 'ty', 'tz']) - obs_z_offset
                point_arrays.append(obs_rear_points)
                filtered_point_arrays.append(filter_outliers(obs_rear_points))
                if len(obs_front_rtk_df.index):
                    obs_z_offset = np.array([0., 0., md['front_gps_h']])
                    rtk_z_offsets.append(obs_z_offset)
                    obs_front_points = obs_front_rtk_df.as_matrix(columns=['tx', 'ty', 'tz']) - obs_z_offset
                    point_arrays.append(obs_front_points)
                    filtered_point_arrays.append(filter_outliers(obs_front_points))

        if correct == CORRECT_PLANE:
            points = np.array(np.concatenate(filtered_point_arrays))
            centroid, normal, rotation = fit_plane(
                points, do_plot=True, dataset_outdir=dataset_outdir, name=bs.name)

            def apply_correction(p, z):
                p -= centroid
                p = np.dot(rotation, p.T).T
                c = np.concatenate([centroid[0:2], z[2:]])
                p += c
                return p

            corrected_points = [apply_correction(pa, z) for pa, z in zip(point_arrays, rtk_z_offsets)]
            cap_front_rtk_df.loc[:, ['tx', 'ty', 'tz']] = corrected_points[0]
            cap_rear_rtk_df.loc[:, ['tx', 'ty', 'tz']] = corrected_points[1]
            pts_idx = 2
            for obs_name in obs_rtk_df_dict.keys():
                obs_rtk_df_dict[obs_name]['rear'].loc[:, ['tx', 'ty', 'tz']] = corrected_points[pts_idx]
                pts_idx += 1
                if 'front' in obs_rtk_df_dict[obs_name]:
                    obs_rtk_df_dict[obs_name]['front'].loc[:, ['tx', 'ty', 'tz']] = corrected_points[pts_idx]
                    pts_idx += 1

        if len(camera_dict['timestamp']):
            # Interpolate samples from all used sensors to camera frame timestamps
            camera_df['timestamp'] = pd.to_datetime(camera_df['timestamp'])
            camera_df.set_index(['timestamp'], inplace=True)
            camera_df.index.rename('index', inplace=True)
            camera_index_df = pd.DataFrame(index=camera_df.index)

            def interpolate_and_save(df, cols, filename):
                interp_df = interpolate_to_camera(camera_index_df, df, filter_cols=cols)
                interp_df.to_csv(os.path.join(dataset_outdir, filename), header=True)
                return interp_df

            interpolate_and_save(
                cap_rear_gps_df, gps_cols, 'capture_vehicle_rear_gps_interp.csv')
            interpolate_and_save(
                cap_front_gps_df, gps_cols, 'capture_vehicle_front_gps_interp.csv')
            cap_rear_rtk_interp = interpolate_and_save(
                cap_rear_rtk_df, rtk_cols, 'capture_vehicle_rear_rtk_interp.csv')
            cap_front_rtk_interp = interpolate_and_save(
                cap_front_rtk_df, rtk_cols, 'capture_vehicle_front_rtk_interp.csv')

            if not obs_rtk_df_dict:
                print('Warning: No obstacles or obstacle RTK data present. '
                      'Skipping Tracklet generation for %s.' % bs.name)
                continue
            if not bs.metadata:
                print('Error: No metadata found, metadata.csv file should be with .bag files.'
                      'Skipping tracklet generation.')
                continue

            cap_front_rtk_rec = cap_front_rtk_interp.to_dict(orient='records')
            cap_rear_rtk_rec = cap_rear_rtk_interp.to_dict(orient='records')
            collection = TrackletCollection()
            for obs_name in obstacle_rtk_dicts.keys():
                obs_rear_interp = interpolate_and_save(
                    obs_rtk_df_dict[obs_name]['rear'], rtk_cols, '%s_rear_rtk_interpolated.csv' % obs_name)
                obs_rear_rec = obs_rear_interp.to_dict(orient='records')
                if 'front' in obs_rtk_df_dict[obs_name]:
                    obs_front_interp = interpolate_and_save(
                        obs_rtk_df_dict[obs_name]['front'], rtk_cols, '%s_rear_rtk_interpolated.csv' % obs_name)
                    obs_front_rec = obs_front_interp.to_dict(orient='records')
                else:
                    obs_front_rec = {}

                # Plot obstacle and front/rear rtk paths in absolute RTK ENU coords
                fig = plt.figure()
                plt.plot(
                    obs_rear_interp['tx'].tolist(),
                    obs_rear_interp['ty'].tolist(),
                    cap_front_rtk_interp['tx'].tolist(),
                    cap_front_rtk_interp['ty'].tolist(),
                    cap_rear_rtk_interp['tx'].tolist(),
                    cap_rear_rtk_interp['ty'].tolist())
                if 'front' in obs_rtk_df_dict[obs_name]:
                    plt.plot(
                        obs_front_interp['tx'].tolist(),
                        obs_front_interp['ty'].tolist())
                fig.savefig(os.path.join(dataset_outdir, '%s-%s-plot.png' % (bs.name.replace('/', '-'), obs_name)))
                plt.close(fig)

                # Extract lwh and object type from CSV metadata mapping file
                md = extract_metadata(bs.metadata, obs_name)

                obs_tracklet = Tracklet(
                    object_type=md['object_type'], l=md['l'], w=md['w'], h=md['h'], first_frame=0)

                # NOTE these calculations are done in obstacle oriented coordinates. The LWH offsets from
                # metadata specify offsets from lower left, rear, ground corner of the vehicle. Where +ve is
                # along the respective length, width, height axis away from that point. They are converted to
                # velodyne/ROS compatible X,Y,Z where X +ve is forward, Y +ve is left, and Z +ve is up.
                lrg_to_centroid = [md['l'] / 2., -md['w'] / 2., md['h'] / 2.]
                if 'front' in obs_rtk_df_dict[obs_name]:
                    lrg_to_front_gps = [md['front_gps_l'], -md['front_gps_w'], md['front_gps_h']]
                    gps_to_centroid = np.subtract(lrg_to_centroid, lrg_to_front_gps)
                else:
                    lrg_to_rear_gps = [md['rear_gps_l'], -md['rear_gps_w'], md['rear_gps_h']]
                    gps_to_centroid = np.subtract(lrg_to_centroid, lrg_to_rear_gps)

                # From capture vehicle 'GPS FRONT' - 'LIDAR' in
                # https://github.com/udacity/didi-competition/blob/master/mkz-description/mkz.urdf.xacro
                velo_to_front = [-1.0922, 0, -0.0508]

                # This would be better handled by more accurate GPS unit mounting/measurement on capture vehicle
                yaw_err_rad = yaw_err * np.pi / 180
                pitch_err_rad = pitch_err * np.pi / 180

                # Convert ENU RTK coords of obstacle to capture vehicle body frame relative coordinates
                if obs_front_rec:
                    rtk_coords = zip(cap_front_rtk_rec, cap_rear_rtk_rec, obs_front_rec, obs_rear_rec)
                    obs_tracklet.poses = [get_obstacle_pose(
                        c[0], c[1], c[2], c[3],
                        gps_to_centroid, velo_to_front, yaw_err_rad, pitch_err_rad) for c in rtk_coords]
                else:
                    rtk_coords = zip(cap_front_rtk_rec, cap_rear_rtk_rec, obs_rear_rec)
                    obs_tracklet.poses = [get_obstacle_pose(
                        c[0], c[1], {}, c[2],
                        gps_to_centroid, velo_to_front, yaw_err_rad, pitch_err_rad) for c in rtk_coords]

                collection.tracklets.append(obs_tracklet)
                # end for obs_topic loop

            tracklet_path = os.path.join(dataset_outdir, 'tracklet_labels.xml')
            collection.write_xml(tracklet_path)
        else:
            print('Warning: No camera image times were found. '
                  'Skipping sensor interpolation and Tracklet generation.')


if __name__ == '__main__':
    main()
