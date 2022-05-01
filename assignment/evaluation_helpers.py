import numpy as np
import scipy
from shapely.geometry import Point
from tensorflow.image import non_max_suppression
from tqdm.notebook import tqdm

from practicum1.BoundingBox import BoundingBox
from practicum1.ImagePatch import ImagePatch

from ipynb.fs.defs.fa_01a_data_visualization import get_bounding_box_from_object

def get_gt_bboxes(sequence):
    gt_bboxes = [
        # [(v_min, u_min, v_max, u_max),],  # replace me with lists of tuples for each frame within the sequence
    ]

    # create ground truth boxes (copied over from practicum1)

    # TODO (low prio) refactor to be without intermediate BoundingBox representation
    sequence_labels = []
    for frame_data in sequence:
        sequence_labels.append(frame_data.get_labels_camera())

    sequence_bboxes = []
    for frame_labels in sequence_labels:
        frame_bboxes = []
        for label in frame_labels:
            if label["label_class"] == "Pedestrian":
                bbox_coords = label["2d_bbox"]
                v1 = bbox_coords["y1"]
                v2 = bbox_coords["y2"]
                u1 = bbox_coords["x1"]
                u2 = bbox_coords["x2"]

                bbox = BoundingBox(v1, u1, v2, u2, from_corners=True)
                frame_bboxes.append(bbox)
        sequence_bboxes.append(frame_bboxes)

    gt_bboxes = [[bbox.get_bbox_corners() for bbox in frame_bboxes] for frame_bboxes in sequence_bboxes]
    return gt_bboxes


def get_sequence_proposals(sequence, frame_pedestrian_dicts):
    # generate sequence_proposals from frame_pedestrian_dicts
    sequence_proposals = []  # List[List[ImagePatch]]

    for measurements, (frame_index, pedestrian_dicts) in zip(sequence, frame_pedestrian_dicts.items()):
        camera_projection_matrix = measurements.get_camera_projection_matrix()
        frame_proposals = []
        for pedestrian_dict in pedestrian_dicts:
            bbox = get_bounding_box_from_object(pedestrian_dict, camera_projection_matrix)
            # make one-element list to fulfull API for generate_metrics_dict
            score = np.asarray(
                [
                    pedestrian_dict["score"],
                ]
            )
            image_patch = ImagePatch(None, bbox, score)
            frame_proposals.append(image_patch)
        sequence_proposals.append(frame_proposals)
    return sequence_proposals


def IOU_circle(point1, point2, radius):
    x1, y1 = point1['x'], point1['y']
    x2, y2 = point2['x'], point2['y']

    # create polygons
    polygon1 = Point(x1, y1).buffer(radius)  # create circle with radius (represented as polygon)
    polygon2 = Point(x2, y2).buffer(radius)  # create circle with radius (represented as polygon)

    # calculate areas
    intersection_area = polygon1.intersection(polygon2).area
    union_area = polygon1.union(polygon2).area
    
    assert intersection_area >= 0.0
    assert union_area >= 0

    # calculate IOU
    IOU = intersection_area / union_area
    
    assert IOU <= 1.
    assert IOU >= 0.

    return IOU


## Copy from BoundingBox.py and change for circle
# TODO refactor to avoid code duplication, i.e. add IOU_function pointer as parameter
def match_bboxes_circle(bbox_gt, bbox_pred, IOU_THRESH=0.0, radius=None):
    '''
    Given sets of true and predicted bounding-boxes,
    determine the best possible match.
    Parameters
    ----------
    bbox_gt, bbox_pred : N1x4 and N2x4 np array of bboxes [x1,y1,x2,y2]. 
      The number of bboxes, N1 and N2, need not be the same.
    
    Returns
    -------
    (idxs_true, idxs_pred, ious, label_val, gt_assigned)
        idxs_true, idxs_pred : indices into gt and pred for matches
        ious : corresponding IOU value of each match
        label_val: vector of 0/1 values for the list of detections
        gt_assigned: vector of 0/1 for the list of GTs
    '''
    n_true = bbox_gt.shape[0]
    n_pred = bbox_pred.shape[0]
    MAX_DIST = 1.0
    MIN_IOU = 0.0

    # NUM_GT x NUM_PRED
    iou_matrix = np.zeros((n_true, n_pred))
    for i in range(n_true):
        for j in range(n_pred):
            iou_matrix[i, j] = IOU_circle(bbox_gt[i], bbox_pred[j], radius=radius)

    if n_pred > n_true:
      # there are more predictions than ground-truth - add dummy rows
        diff = n_pred - n_true
        iou_matrix = np.concatenate( (iou_matrix, 
                                    np.full((diff, n_pred), MIN_IOU)), 
                                  axis=0)

    if n_true > n_pred:
      # more ground-truth than predictions - add dummy columns
        diff = n_true - n_pred
        iou_matrix = np.concatenate( (iou_matrix, 
                                    np.full((n_true, diff), MIN_IOU)), 
                                  axis=1)

    # call the Hungarian matching
    idxs_true, idxs_pred = scipy.optimize.linear_sum_assignment(1 - iou_matrix)

    if (not idxs_true.size) or (not idxs_pred.size):
        ious = np.array([])
    else:
        ious = iou_matrix[idxs_true, idxs_pred]

    # remove dummy assignments
    sel_pred = idxs_pred<n_pred
    idx_pred_actual = idxs_pred[sel_pred] 
    idx_gt_actual = idxs_true[sel_pred]
    ious_actual = iou_matrix[idx_gt_actual, idx_pred_actual]
    sel_valid = (ious_actual > IOU_THRESH)
    label_val = sel_valid.astype(int)

    # check for assigned GTs
    gt_assigned = np.asarray([i in idx_gt_actual[sel_valid] for i in range(n_true)], dtype=int)

    return idx_gt_actual[sel_valid], idx_pred_actual[sel_valid], ious_actual[sel_valid], label_val, gt_assigned


# Copy from practicum1/evaluation.py and change for circle instead of bbox
# TODO refactor to avoid code duplication
def match_bboxes_and_calc_precision_and_recall_circle(gt_bboxes, pred_bboxes, iou_thresh=0.5, radius=None): 
    num_true = len(gt_bboxes)
    num_preds = len(pred_bboxes) 
    
    assignments = match_bboxes_circle(gt_bboxes, pred_bboxes, IOU_THRESH=iou_thresh, radius=radius)

    gt_assignments = assignments[4]
    assignment_ious = assignments[2]
    true_positives = 0
    false_negatives = 0

    for gt_assigned in gt_assignments:
        if gt_assigned:
            true_positives += 1
        else:
            false_negatives += 1

    false_positives = num_preds - true_positives 
    
    all_positives = float(true_positives + false_negatives)
    if all_positives == 0:
        recall = 0.
    else:
        recall = float(true_positives) / all_positives 

    if false_positives == 0:
        precision = 1.
    else:
        precision = float(true_positives) / float(num_preds) 
    
    return precision, recall
    

# Copy from practicum1/evaluation.py and change for circle instead of bbox
# TODO refactor to avoid code duplication
def generate_metrics_dict_circle(sequence_proposals, gt_points, discrimination_thresholds, iou_thresholds, radius):
    """
    sequence_proposals: List[List[ImagePatch]]  # patch.bbox and patch.score[0] being used
    gt_bboxes: List[List[Tuple(v1, u1, v2, u2)]]
    discrimination_thresholds: List[float]
    iou_thresholds: List[float]
    radius: float radius in meters to consider for circle IOU
    """
    metrics = []
    for thresh in tqdm(discrimination_thresholds):
        # Classifier discrimination threshold
        pred_points = [[proposal_pnt for proposal_pnt in frame if proposal_pnt['score']>=thresh] for frame in sequence_proposals] #CHANGE

        frame_metrics = []
            
        assert len(sequence_proposals) == len(gt_points), (len(sequence_proposals), len(gt_points))
        assert len(sequence_proposals) == len(pred_points)

        for i in range(len(sequence_proposals)):
            frame_gt_points = np.asarray(gt_points[i])
            frame_pred_points = np.asarray(pred_points[i])

            iou_thresh_metrics = []
            for iou_thresh in iou_thresholds:
                # IoU matching threshold
                precision, recall = match_bboxes_and_calc_precision_and_recall_circle(frame_gt_points,
                                                                                      frame_pred_points,
                                                                                      iou_thresh=iou_thresh,
                                                                                      radius=radius)
                iou_thresh_metrics.append([precision, recall])

            frame_metrics.append(iou_thresh_metrics)
        metrics.append(np.mean(frame_metrics, axis=0))

    metrics = np.asarray(metrics)
    # 'metrics' has shape (len(discrimination_thresholds), len(iou_thresholds), 2)
    metrics = np.swapaxes(metrics, 0, 1)
    # 'metrics' has shape (len(iou_thresholds), len(discrimination_thresholds), 2)
    metrics_dict = {iou_thresholds[i]: metrics[i] for i in range(metrics.shape[0])}
    
    return metrics_dict


def get_sequence_proposals_circle(frame_ped_positions, frame_ped_scores):
    sequence__proposals = []
    for ped_positions, ped_scores in zip(frame_ped_positions.values(), frame_ped_scores.values()):
        frame__proposals = []
        for ped_position, ped_score in zip(ped_positions, ped_scores):
            frame__proposals.append({
                'x': ped_position[0],
                'y': ped_position[1],
                'score': ped_score
            })
        sequence__proposals.append(frame__proposals)
    return sequence__proposals


def get_GT_sequence_groundplane_proposals(frame_ped_gts):
    GT_sequence_groundplane_proposals = []
    for ped_gts in frame_ped_gts.values():

        frame_groundplane_proposals = []
        for ped_gt in ped_gts:
             frame_groundplane_proposals.append({'x': ped_gt[0],
                                                 'y': ped_gt[1]
                                                })
        GT_sequence_groundplane_proposals.append(frame_groundplane_proposals)
    return GT_sequence_groundplane_proposals
