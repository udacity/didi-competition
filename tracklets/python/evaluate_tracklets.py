#! /usr/bin/python
""" Udacity Self-Driving Car Challenge Tracklet evaluation script
"""

from __future__ import print_function, division
from shapely.geometry import Polygon
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import argparse
import os
import sys
import yaml
import math
import glob

from parse_tracklet import *

VOLUME_METHODS = ['box', 'cylinder', 'sphere']
CLASS_WEIGHTING = ['volume', 'instance', 'simple', 'none']


def lwh_to_box(l, w, h):
    box = np.array([
        [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
        [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2],
    ])
    return box


def intersect_bbox_with_yaw(box_a, box_b):
    """
    A simplified calculation of 3d bounding box intersection.
    It is assumed that the bounding box is only rotated
    around Z axis (yaw) from an axis-aligned box.

    :param box_a, box_b: obstacle bounding boxes for comparison
    :return: intersection volume (float)
    """
    # height (Z) overlap
    min_h_a = np.min(box_a[2])
    max_h_a = np.max(box_a[2])
    min_h_b = np.min(box_b[2])
    max_h_b = np.max(box_b[2])
    max_of_min = np.max([min_h_a, min_h_b])
    min_of_max = np.min([max_h_a, max_h_b])
    z_intersection = np.max([0, min_of_max - max_of_min])
    if z_intersection == 0:
        return 0.

    # oriented XY overlap
    xy_poly_a = Polygon(zip(*box_a[0:2, 0:4]))
    xy_poly_b = Polygon(zip(*box_b[0:2, 0:4]))
    xy_intersection = xy_poly_a.intersection(xy_poly_b).area
    if xy_intersection == 0:
        return 0.

    return z_intersection * xy_intersection


def intersect_sphere(sphere_a, sphere_b):
    """
    3d sphere intersection.

    :param sphere_a, sphere_b: spheres for comparison
        packed as x,y,z,r, xyz points to center of sphere
    :return: intersection volume (float)
    """
    assert len(sphere_a) == len(sphere_b) == 4
    dist = np.linalg.norm(sphere_a[0:3] - sphere_b[0:3])
    r_a, r_b = sphere_a[3], sphere_b[3]
    if dist >= r_a + r_b:
        # spheres do not overlap in any way
        return 0.
    elif dist <= abs(r_a - r_b):
        # one sphere fully inside the other (includes coincident)
        # take volume of smaller sphere as intersection
        intersection = 4/3. * np.pi * min(r_a, r_b)**3
    else:
        # spheres partially overlap, calculate intersection as per
        # http://mathworld.wolfram.com/Sphere-SphereIntersection.html
        intersection = (r_a + r_b - dist)**2
        intersection *= (dist**2 + 2*dist*(r_a + r_b) - 3*(r_a - r_b)**2)
        intersection *= np.pi / (12*dist)
    return intersection


def intersect_cylinder(cyl_a, cyl_b):
    """
    3d cylinder intersection.

    :param cyl_a, cyl_b: cylinders for comparison
        packed as x,y,z,r,h, xyz points to center of axis aligned (h along z) cylinder
    :return: intersection volume (float)
    """
    assert len(cyl_a) == len(cyl_b) == 5
    xyz_a, xyz_b = cyl_a[0:3], cyl_b[0:3]
    r_a, r_b = cyl_a[3], cyl_b[3]
    h_a, h_b = cyl_a[4], cyl_b[4]

    # height (Z) overlap
    zh_a = [xyz_a[2] + h_a/2, xyz_a[2] - h_a/2]
    zh_b = [xyz_b[2] + h_b/2, xyz_b[2] - h_b/2]
    max_of_min = np.max([zh_a[1], zh_b[1]])
    min_of_max = np.min([zh_a[0], zh_b[0]])
    z_intersection = np.max([0, min_of_max - max_of_min])
    if z_intersection == 0:
        return 0.

    dist = np.linalg.norm(cyl_a[0:2] - cyl_b[0:2])
    if dist >= r_a + r_b:
        # cylinders do not overlap in any way
        return 0.
    elif dist <= abs(r_a - r_b):
        # one cylinder fully inside the other (includes coincident)
        # take volume of smaller sphere as intersection
        intersection = np.pi * min(r_a, r_b)**2 * z_intersection
    else:
        def _lens(r, ro, d):
            td = (d**2 + r**2 - ro**2)/(2*d)
            return r**2 * math.acos(td/r) - td * math.sqrt(r**2 - td**2)
        circle_intersection = _lens(r_a, r_b, dist) + _lens(r_b, r_a, dist)
        intersection = circle_intersection * z_intersection
    return intersection


def iou(vol_a, vol_b, vol_intersect):
    union = vol_a + vol_b - vol_intersect
    return vol_intersect / union if union else 0.


def dice(vol_a, vol_b, vol_intersect):
    return 2 * vol_intersect / (vol_a + vol_b) if (vol_a + vol_b) else 0.


class Obs(object):

    def __init__(self, tracklet_idx, object_type, size, position, rotation):
        self.tracklet_idx = tracklet_idx
        self.object_type = object_type
        self.h, self.w, self.l = np.clip(size, a_min=0., a_max=100.)
        self.position = position
        self.yaw = rotation[2]
        self._oriented_bbox = None  # for caching

    def get_bbox(self):
        if self._oriented_bbox is None:
            bbox = lwh_to_box(self.l, self.w, self.h)
            # calc 3D bound box in capture vehicle oriented coordinates
            rot_mat = np.array([
                [np.cos(self.yaw), -np.sin(self.yaw), 0.0],
                [np.sin(self.yaw), np.cos(self.yaw), 0.0],
                [0.0, 0.0, 1.0]])
            self._oriented_bbox = np.dot(rot_mat, bbox) + np.tile(self.position, (8, 1)).T
        return self._oriented_bbox

    def get_sphere(self):
        # For a quick and dirty bounding sphere we will take 1/2 the largest
        # obstacle dim as the radius and call it close enough, for our purpose won't
        # make a noteworthy difference from a circumscribed sphere of the bbox
        r = max(self.h, self.w, self.l)/2
        # sphere passed as 4 element vector with radius as last element
        return np.append(self.position, r)

    def get_cylinder(self):
        r = max(self.w, self.l)/2
        # cylinder passed as 5 element vector with radius and height as last elements
        return np.append(self.position, [r, self.h])

    def get_vol_sphere(self):
        r = max(self.h, self.w, self.l)/2
        return 4/3. * np.pi * r**3

    def get_vol_cylinder(self):
        r = max(self.w, self.l)/2
        return np.pi * r**2 * self.h

    def get_vol_box(self):
        return self.h * self.w * self.l

    def get_vol(self, method='box'):
        if method == 'sphere':
            return self.get_vol_sphere()
        elif method == 'cylinder':
            return self.get_vol_cylinder()
        return self.get_vol_box()

    def intersection_metric(self, other, method='box', metric_fn=iou):
        if method == 'sphere':
            intersection_vol = intersect_sphere(self.get_sphere(), other.get_sphere())
            metric_val = metric_fn(self.get_vol_sphere(), other.get_vol_sphere(), intersection_vol)
        elif method == 'cylinder':
            intersection_vol = intersect_cylinder(self.get_cylinder(), other.get_cylinder())
            metric_val = metric_fn(self.get_vol_cylinder(), other.get_vol_cylinder(), intersection_vol)
        else:
            intersection_vol = intersect_bbox_with_yaw(self.get_bbox(), other.get_bbox())
            metric_val = metric_fn(self.get_vol_box(), other.get_vol_box(), intersection_vol)
        return metric_val, intersection_vol

    def __repr__(self):
        return str(self.tracklet_idx) + ' ' + str(self.object_type)


def get_vol_method(object_type, override=''):
    if override and override in VOLUME_METHODS:
        return override
    if object_type == 'Pedestrian':
        return 'cylinder'
    else:
        return 'box'


def generate_obstacles(tracklets, override_size=None):
    for tracklet_idx, tracklet in enumerate(tracklets):
        frame_idx = tracklet.first_frame
        for trans, rot in zip(tracklet.trans, tracklet.rots):
            obstacle = Obs(
                tracklet_idx,
                tracklet.object_type,
                override_size if override_size is not None else tracklet.size,
                trans,
                rot)
            yield frame_idx, obstacle
            frame_idx += 1


class EvalFrame(object):

    def __init__(self):
        self.gt_obs = []
        self.pred_obs = []

    def score(
            self,
            intersection_vol_count,
            combined_vol_count,
            gt_vol_count,
            instance_count,
            pr_at_thresh,
            method_override='',
            metric_fn=iou):
        # Perform IOU/Dice calculations between all gt and predicted obstacle pairings and greedily match those
        # with the largest IOU. Possibly other matching algorithms will work better/be more efficient.
        # NOTE: This is not a tracking oriented matching like MOTA, predicted -> gt affinity context
        # would need to be passed between frame evaluations for that.

        intersections = []
        fn = set(range(len(self.gt_obs)))  # gt idx for gt that don't have any intersecting pred
        fp = set(range(len(self.pred_obs)))  # pred idx for pred not intersecting any gt

        # Compute metric between all obstacle gt <-> prediction pairing possibilities (of same obj type)
        for p_idx, p in enumerate(self.pred_obs):
            for g_idx, g in enumerate(self.gt_obs):
                if p.object_type == g.object_type:
                    method = get_vol_method(p.object_type, method_override)
                    metric_val, intersection_vol = g.intersection_metric(p, method=method, metric_fn=metric_fn)
                    if metric_val > 0:
                        intersections.append((metric_val, intersection_vol, p_idx, g_idx))

        # Traverse calculated intersections, greedily consume intersections with largest overlap first,
        # summing volumes and marking TP/FP/FN at specific IOU thresholds as we go.
        intersections.sort(key=lambda x: x[0], reverse=True)
        for metric_val, intersection_vol, p_idx, g_idx in intersections:
            if g_idx in fn and p_idx in fp:
                fn.remove(g_idx)  # consume the ground truth
                fp.remove(p_idx)  # consume the prediction
                gt_obs, pred_obs = self.gt_obs[g_idx], self.pred_obs[p_idx]
                #print('Metric: ', metric_val, intersection_vol)
                intersection_vol_count[gt_obs.object_type] += intersection_vol
                method = get_vol_method(gt_obs.object_type, method_override)
                gt_vol = gt_obs.get_vol(method)
                gt_vol_count[gt_obs.object_type] += gt_vol
                combined_vol_count[gt_obs.object_type] += gt_vol + pred_obs.get_vol(method)
                instance_count[gt_obs.object_type] += 1
                for metric_threshold in pr_at_thresh.keys():
                    if metric_val > metric_threshold:
                        pr_at_thresh[metric_threshold]['TP'] += 1
                    else:
                        # It's already determined at this point that this is the highest IOU score
                        # there is for this match and another match for these two bbox won't be considered.
                        # Thus both the gt and prediction should be considered unmatched.
                        pr_at_thresh[metric_threshold]['FP'] += 1
                        pr_at_thresh[metric_threshold]['FN'] += 1

        # sum remaining false negative volume (unmatched ground truth box volume)
        for g_idx in fn:
            gt_obs = self.gt_obs[g_idx]
            method = get_vol_method(gt_obs.object_type, method_override)
            gt_vol = gt_obs.get_vol(method)
            gt_vol_count[gt_obs.object_type] += gt_vol
            combined_vol_count[gt_obs.object_type] += gt_vol
            instance_count[gt_obs.object_type] += 1
            for metric_threshold in pr_at_thresh.keys():
                pr_at_thresh[metric_threshold]['FN'] += 1

        # sum remaining false positive volume (unmatched prediction volume)
        for p_idx in fp:
            pred_obs = self.pred_obs[p_idx]
            method = get_vol_method(pred_obs.object_type, method_override)
            combined_vol_count[pred_obs.object_type] += pred_obs.get_vol(method)
            for metric_threshold in pr_at_thresh.keys():
                pr_at_thresh[metric_threshold]['FP'] += 1


def load_indices(indices_file):
    with open(indices_file, 'r') as f:
        def safe_int(x):
            try:
                return int(x.split(',')[0])
            except ValueError:
                return 0
        indices = [safe_int(line) for line in f][1:]  # skip header line
    return indices


def process_sequence(
        seq_files,
        counters,
        pr_at_thresh,
        metric_fn,
        process_params,
):
    override_volume_method = process_params['override_volume_method']
    override_lwh_with_gt = process_params['override_lwh_with_gt']

    print('Processing sequence with:')
    print('\tground-truth file: %s' % seq_files['gt_file'])
    if 'pred_file' in seq_files:
        print('\tprediction file: %s' % seq_files['pred_file'])
    if 'include_indices_file' in seq_files:
        print('\tinclude indices file: %s' % seq_files['include_indices_file'])
    if 'exclude_indices_file' in seq_files:
        print('\texclude indices file: %s' % seq_files['exclude_indices_file'])

    if 'pred_file' in seq_files:
        pred_tracklets = parse_xml(seq_files['pred_file'])
        if not pred_tracklets:
            sys.stderr.write('Error: No Tracklets parsed for predictions.\n')
            exit(-1)
    else:
        pred_tracklets = []

    gt_tracklets = parse_xml(seq_files['gt_file'])
    if not gt_tracklets:
        sys.stderr.write('Error: No Tracklets parsed for ground truth.\n')
        exit(-1)

    num_gt_frames = 0
    for gt_tracklet in gt_tracklets:
        num_gt_frames = max(num_gt_frames, gt_tracklet.first_frame + gt_tracklet.num_frames)

    num_pred_frames = 0
    for p_idx, pred_tracklet in enumerate(pred_tracklets):
        num_pred_frames = max(num_pred_frames, pred_tracklet.first_frame + pred_tracklet.num_frames)
        if process_params['test_mode']:
            trans_noise = np.random.normal(0, 0.3, pred_tracklet.trans.shape)
            rots_noise = np.random.normal(0, 0.39, pred_tracklet.rots.shape)
            pred_tracklets[p_idx].trans = pred_tracklet.trans + trans_noise
            pred_tracklets[p_idx].rots = pred_tracklet.rots + rots_noise
            pred_tracklets[p_idx].size = pred_tracklet.size + np.random.normal(0, 0.2, pred_tracklet.size.shape)

    num_frames = max(num_gt_frames, num_pred_frames)
    if not num_frames:
        print('Error: No frames to evaluate')
        exit(-1)

    if 'include_indices_file' in seq_files:
        if not os.path.exists(seq_files['include_indices_file']):
            sys.stderr.write('Error: Filter indices files specified but does not exist.\n')
            exit(-1)
        eval_indices = load_indices(seq_files['include_indices_file'])
        print('Filter file %s loaded with %d frame indices to include for evaluation.' %
              (seq_files['include_indices_file'], len(eval_indices)))
    else:
        eval_indices = list(range(num_frames))

    if 'exclude_indices_file' in seq_files:
        if not os.path.exists(seq_files['exclude_indices_file']):
            sys.stderr.write('Error: Exclude indices files specified but does not exist.\n')
            exit(-1)
        exclude_indices = set(load_indices(seq_files['exclude_indices_file']))
        eval_indices = [x for x in eval_indices if x not in exclude_indices]
        print('Exclude file %s loaded with %d frame indices to exclude from evaluation.' %
              (seq_files['exclude_indices_file'], len(exclude_indices)))

    eval_frames = {i: EvalFrame() for i in eval_indices}

    included_gt = 0
    excluded_gt = 0
    for frame_idx, obstacle in generate_obstacles(gt_tracklets):
        if frame_idx in eval_frames:
            eval_frames[frame_idx].gt_obs.append(obstacle)
            included_gt += 1
        else:
            excluded_gt += 1

    included_pred = 0
    excluded_pred = 0
    gt_size = gt_tracklets[0].size if override_lwh_with_gt else None
    for frame_idx, obstacle in generate_obstacles(pred_tracklets, override_size=gt_size):
        if frame_idx in eval_frames:
            eval_frames[frame_idx].pred_obs.append(obstacle)
            included_pred += 1
        else:
            excluded_pred += 1

    print('%d ground truth object instances included in evaluation, % s excluded' % (included_gt, excluded_gt))
    print('%d predicted object instances included in evaluation, % s excluded' % (included_pred, excluded_pred))

    for frame_idx in eval_indices:
        #  calc scores
        eval_frames[frame_idx].score(
            counters['intersection_volume'],
            counters['combined_volume'],
            counters['gt_volume'],
            counters['gt_instance_count'],
            pr_at_thresh,
            method_override=override_volume_method,
            metric_fn=metric_fn)


def process_submission(
        pred_path,
        gt_path,
        include_indices_path,
        exclude_indices_path,
        process_params,
        metric_params,
        output_dir,
        class_weighting='instance',
        prefix='',
        suppress_print=False):

    sequences = []
    if os.path.isfile(pred_path) and os.path.isfile(gt_path):
        seq = {'gt_file': gt_path, 'pred_file': pred_path}
        if include_indices_path and os.path.isfile(include_indices_path):
            seq['include_indices_file'] = include_indices_path
        if exclude_indices_path and os.path.isfile(exclude_indices_path):
            seq['exclude_indices_file'] = exclude_indices_path
        sequences.append(seq)
    elif os.path.isdir(pred_path) and os.path.isdir(gt_path):
        def _f(path):
            return os.path.splitext(os.path.basename(path))[0]
        pred_files = {_f(x): x for x in glob.glob(os.path.join(pred_path, '*.xml'))}
        include_indices_files = {}
        if include_indices_path and os.path.isdir(include_indices_path):
            include_indices_files = {_f(x): x for x in glob.glob(os.path.join(include_indices_path, '*.csv'))}
        exclude_indices_files = {}
        if exclude_indices_path and os.path.isdir(exclude_indices_path):
            exclude_indices_files = {_f(x): x for x in glob.glob(os.path.join(exclude_indices_path, '*.csv'))}
        gt_files = glob.glob(os.path.join(gt_path, '*.xml'))
        for g in gt_files:
            pb = _f(g)
            seq = {'gt_file': g}
            if pb in pred_files:
                seq['pred_file'] = pred_files[pb]
            if pb in include_indices_files:
                seq['include_indices_file'] = include_indices_files[pb]
            if pb in exclude_indices_files:
                seq['exclude_indices_file'] = exclude_indices_files[pb]
            sequences.append(seq)
        if len(pred_files) < len(gt_files):
            print('Warning: Only %d of %d ground-truth files matched with predictions.'
                  % (len(pred_files), len(gt_files)))
        assert len(gt_files) == len(sequences)

    else:
        sys.stderr.write('Error: Ground-truth and predicted paths must both be files or both be folders.\n')
        exit(-1)

    if not sequences:
        print('Error: No predicted sequences were found to evaluate.')
        exit(-1)

    metric_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    pr_at_thresh = {k: Counter() for k in metric_thresholds}
    counters = defaultdict(Counter)
    for s in sequences:
        process_sequence(s, counters, pr_at_thresh, metric_params['fn'], process_params)

    # Class weighting
    #  * '' or 'none': calculate the metric across all class volumes (large volume objects dominate small)
    #  * 'simple': simple mean of per class metric values
    #  * 'instance': ground-truth instance count weighted mean of per class metric values
    #  * 'volume': ground-truth volume weighted mean of per class metric values
    metric_value_sum = 0.0
    metric_weight_sum = 0.0
    combined_vol_sum = 0.0
    intersection_vol_sum = 0.0
    results_table = {metric_params['metric_per_obj']: {}, metric_params['pr_per_metric']: {}}
    for k in counters['combined_volume'].keys():
        combined_vol_sum += counters['combined_volume'][k]
        intersection_vol_sum += counters['intersection_volume'][k]
        metric_val = metric_params['fn'](counters['combined_volume'][k], 0., counters['intersection_volume'][k])
        results_table[metric_params['metric_per_obj']][k] = float(metric_val)
        if class_weighting == 'instance':
            metric_weight = counters['gt_instance_count'][k]
        elif class_weighting == 'volume':
            metric_weight = counters['gt_volume'][k]
        else:
            metric_weight = 1.0  # for 'simple' or 'none'
        metric_value_sum += metric_val * metric_weight
        metric_weight_sum += metric_weight

    if class_weighting and class_weighting != "none":
        # class weighting is enabled, use the summed metric calculated for appropriate weighting method
        all_metric = metric_value_sum / metric_weight_sum if metric_weight_sum else 0.
    else:
        # no weighting is enabled, compute the metric over the volumes summed across all classes
        all_metric = metric_params['fn'](combined_vol_sum, 0., intersection_vol_sum)
    results_table[metric_params['metric_per_obj']]['All'] = float(all_metric)


    # FIXME add support for per class P/R scores?
    # NOTE P/R scores need further analysis given their use with the greedy pred - gt matching
    for k, v in pr_at_thresh.items():
        p = v['TP'] / (v['TP'] + v['FP']) if v['TP'] else 0.0
        r = v['TP'] / (v['TP'] + v['FN']) if v['TP'] else 0.0
        results_table[metric_params['pr_per_metric']][k] = {'precision': p, 'recall': r}

    if not suppress_print:
        print('\nResults')
        print(yaml.safe_dump(results_table, default_flow_style=False, explicit_start=True))

    if output_dir is not None:
        with open(os.path.join(output_dir, metric_params['metric_per_obj'] + '.csv'), 'w') as f:
            f.write('object_type,%s\n' % metric_params['name'])
            [f.write('{0},{1}\n'.format(k, v))
             for k, v in sorted(results_table[metric_params['metric_per_obj']].items(), key=lambda x: x[0])]
        with open(os.path.join(output_dir, metric_params['pr_per_metric'] + '.csv'), 'w') as f:
            f.write('%s_threshold,p,r\n' % metric_params['name'])
            [f.write('{0},{1},{2}\n'.format(k, v['precision'], v['recall']))
             for k, v in sorted(results_table[metric_params['pr_per_metric']].items(), key=lambda x: x[0])]

    return results_table


def find_submissions(folder):
    submissions = []
    for root, _, files in os.walk(folder, topdown=False):
        for rel_filename in files:
            base, ext = os.path.splitext(rel_filename)
            if ext.lower() == '.xml':
                submissions.append(root)
                break
    return submissions


def get_outdir(base_dir, name=''):
    outdir = os.path.join(base_dir, name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def main():
    parser = argparse.ArgumentParser(description='Evaluate two tracklet files.')
    parser.add_argument('prediction', type=str, nargs='?', default='tracklet_labels.xml',
        help='Predicted tracklet label filename or folder')
    parser.add_argument('groundtruth', type=str, nargs='?', default='tracklet_labels_gt.xml',
        help='Groundtruth tracklet label filename or folder')
    parser.add_argument('-f', '--include_indices', type=str, nargs='?', default=None,
        help='CSV file containing frame indices to include in evaluation. All frames included if argument empty.')
    parser.add_argument('-e', '--exclude_indices', type=str, nargs='?', default=None,
        help='CSV file containing frame indices to exclude (takes priority over inclusions) from evaluation.')
    parser.add_argument('-o', '--outdir', type=str, nargs='?', default=None,
        help='Output folder')
    parser.add_argument('-m', '--method', type=str, nargs='?', default='',
        help='Volume intersection calculation method override. "box", "cylinder", '
             '"sphere" (default = "", no override)')
    parser.add_argument('-v', '--eval_metric', type=str, nargs='?', default='iou',
        help='Eval metric. "iou" or "dice" (default = "iou")')
    parser.add_argument('-w', '--class_weight', type=str, nargs='?', default='instance',
        help='Weighting method across all classes. "simple", "volume", "instance", or "none" (default="instance")')
    parser.add_argument('-g', dest='override_lwh_with_gt', action='store_true',
        help='Override predicted lwh values with value from first gt tracklet.')
    parser.add_argument('--batch', dest='batch_mode', action='store_true', help='Batch mode enable')
    parser.add_argument('--test', dest='test_mode', action='store_true', help='Test mode enable')
    parser.add_argument('-d', dest='debug', action='store_true', help='Debug print enable')
    parser.set_defaults(test_mode=False)
    parser.set_defaults(debug=False)
    parser.set_defaults(override_lwh_with_gt=False)
    args = parser.parse_args()
    include_indices_path = args.include_indices
    exclude_indices_path = args.exclude_indices
    output_dir = args.outdir
    eval_metric = args.eval_metric
    class_weighting = args.class_weight
    if class_weighting not in CLASS_WEIGHTING:
        print('Error: Invalid class weighting "%s". Must be one of %s\n'
              % (class_weighting, CLASS_WEIGHTING))
        exit(-1)

    process_params = dict()
    process_params['test_mode'] = args.test_mode
    process_params['override_lwh_with_gt'] = args.override_lwh_with_gt
    process_params['override_volume_method'] = ''
    if args.method:
        if args.method not in VOLUME_METHODS:
            print('Error: Invalid volume method override "%s". Must be one of %s\n'
                  % (args.method, VOLUME_METHODS))
            exit(-1)
        else:
            print('Overriding volume intersection method with %s' % args.method)
            process_params['override_volume_method'] = args.method

    pred_path = args.prediction
    if not os.path.exists(pred_path):
        sys.stderr.write('Error: Prediction file/folder %s not found.\n' % pred_path)
        exit(-1)

    gt_path = args.groundtruth
    if not os.path.exists(gt_path):
        sys.stderr.write('Error: Ground-truth file/folder %s not found.\n' % gt_path)
        exit(-1)

    metric_params = {}
    if eval_metric == 'dice':
        metric_params['name'] = 'dice'
        metric_params['fn'] = dice
        metric_params['metric_per_obj'] = 'dice_score_per_obj'
        metric_params['pr_per_metric'] = 'pr_per_dice_score'
    else:
        metric_params['name'] = 'iou'
        metric_params['fn'] = iou
        metric_params['metric_per_obj'] = 'iou_per_obj'
        metric_params['pr_per_metric'] = 'pr_per_iou'

    if args.batch_mode:
        results = {}
        submission_paths = find_submissions(pred_path)
        for p in submission_paths:
            pb = os.path.relpath(p, pred_path)
            result = process_submission(
                p, gt_path,
                include_indices_path, exclude_indices_path,
                process_params,
                metric_params,
                get_outdir(output_dir, pb),
                class_weighting=class_weighting,
                suppress_print=True)
            results[pb] = result

        if output_dir:
            per_obj = {}
            print(metric_params['metric_per_obj'])
            for k, r in results.items():
                print(k, os.path.basename(k))
                per_obj[k] = r[metric_params['metric_per_obj']]

            per_obj_df = pd.DataFrame.from_dict(per_obj, orient='index')
            per_obj_df.sort_values('All', ascending=False, inplace=True)
            per_obj_df.to_csv(os.path.join(output_dir, 'results_per_obj.csv'), index=True, index_label='Submission')

        print(yaml.safe_dump(results, default_flow_style=False, explicit_start=True))

    else:
        process_submission(
            pred_path, gt_path,
            include_indices_path, exclude_indices_path,
            process_params, metric_params, output_dir,
            class_weighting=class_weighting)


if __name__ == '__main__':
    main()
