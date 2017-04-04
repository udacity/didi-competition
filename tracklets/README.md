# Tracklet Scripts

This folder contains scripts to create KITTI Tracklet files from rosbag capture data and evaluate Tracklet files. This code will not work on the first dataset release without modification, as the topic names have changed as indicated in the first dataset README.

## Installation

The scripts do not require installation. They can be run in-place or in Docker using included Dockerfile if you do not have a suitable ROS environment setup.

## Usage

If you are planning to use Docker, build the docker container manually or using ./build.sh

### bag_to_kitti.py -- Dump images and create KITTI Tracklet files

For running through Docker, you can use the helper script:
    ./run-bag_to_kitti.sh -i [local dir with folder containing bag file(s)] -o [local output dir] -- [args to pass to python script]

For example, if your dataset bags are in /data/bags/*.bag, and you'd like the output in /output:

    ./run-bag_to_kitti.sh -i /data/bags -o /output

The same as above, but you want to suppress image output:

    ./run-bag_to_kitti.sh -i /data/bags -o /output -- -m
    
Note that when passing paths via -i and -o through Docker THE PATHS MUST BE ABSOLUTE. You cannot map relative paths to Docker.

To run bag_to_kitti.py locally, the same -i -o arguments can be used and additional arguments listed in the help (-h) can also be used directly without passing via --. Any valid path, relative or absolute works when calling the script directly.
    
### evaluate_tracklets.py -- Evaluate predicted Tracklet against ground truth

The evaluate_tracklets script does not depend on a ROS environment so it's less relevant to run in the Docker environment.

Usage is straightforward. Run the script as per

    python evaluate_tracklets.py predicted_tracklet.xml ground_truth_tracklet.xml -o /output/metrics
    
If you don't want metrics output as csv files omit the -o argument.

## Metrics and Scoring

The Tracklet evaluation script currently produces two sets of metrics -- Intersection-over-union calculated on bounding box volumes, and precision and recall for detections evaluated at specific IOU thresholds. A description of each follows. 

For competition scoring, only the IOU metric for 'All' object types relevant to the round being scored will be used. This value can be extracted from the YAML encoded results output to stdout:

    iou_per_obj:
        All: <value>

If metric file output is enabled with the '-o' option, the score can alternatively be extracted from the row with object_type = 'All' in the 'iou_per_obj.csv' file.

### IOU Per Object Type

This is a volume based intersection over union metric. The intersection is the overlapping volume of prediction bounding boxes with ground truth bounding boxes. The union is the combined volume of predicted and ground truth boxes. The IOU is equivalent to TP/(TP + FP + FN) where
 * True positives = correctly predicted volume that overlaps ground truth
 * False positives = incorrectly predicted volume that does not overlap ground truth
 * False negatives = ground truth volume not overlapped by any predictions

For this implementation, all of the intersection and union volumes are added up across all frames for each object type and then the ratio is calculated on the summed volumes. For the 'All' field, the scores across all relevant object types are averaged.

It should be noted that predictions are matched with ground truth boxes of the same object type based on the largest overlap and then neither are matched again in that frame. Unmatched predictions overlapping a ground truth box that's already been matched will be considered false positives. Unmatched ground truth volumes that overlap with predictions better matched with another ground truth will be considered false negatives.  


### Detection Precision and Recall at IOU

In addition to the IOU score, the evaluation script also outputs a set of precision and recall values for detections at different IOU thresholds. A detection, true positive, occurs when a predicted volume overlaps a ground truth volume of the same object type with an IOU value greater than the threshold IOU.

This metric is not used in scoring.
