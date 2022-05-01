

import numpy as np
from tqdm.notebook import tqdm
from tensorflow.image import non_max_suppression

from BoundingBox import match_bboxes


def threshold_predictions(preds, threshold=0.5):
    preds_labels = np.greater(preds, threshold)
    return preds_labels


def match_bboxes_and_calc_precision_and_recall(gt_bboxes, pred_bboxes, iou_thresh=0.5): 
    num_true = len(gt_bboxes)
    num_preds = len(pred_bboxes) 
    
    assignments = match_bboxes(gt_bboxes, pred_bboxes, IOU_THRESH=iou_thresh)

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
    

def generate_metrics_dict(sequence_proposals, gt_bboxes, discrimination_thresholds, iou_thresholds):
    """
    sequence_proposals: List[List[ImagePatch]]  # patch.bbox and patch.score[0] being used
    gt_bboxes: List[List[Tuple(v1, u1, v2, u2)]]
    discrimination_thresholds: List[float]
    iou_thresholds: List[float]
    """
    metrics = []
    for thresh in tqdm(discrimination_thresholds):
        # Classifier discrimination threshold
        pedestrian_patches = [[proposal_patch for proposal_patch in frame if proposal_patch.score >= thresh] for frame in sequence_proposals]

        # Apply NMS - this is part of the model, not the metric calculation
        nms_patches = []
        overlap_thresh = 0.
        for frame in pedestrian_patches:
            all_bboxes = np.asarray([patch.bbox.get_bbox_corners() for patch in frame])
            confidences = np.asarray([patch.score[0] for patch in frame])
            if len(all_bboxes) > 0:
                idx = non_max_suppression(all_bboxes, confidences, max_output_size=len(all_bboxes), iou_threshold=overlap_thresh)
                nms_patches.append([frame[i] for i in idx])
            else:
                nms_patches.append([])

        frame_metrics = []
        pred_bboxes = [[frame.bbox.get_bbox_corners() for frame in frame_patches] for frame_patches in nms_patches]

        assert len(sequence_proposals) == len(gt_bboxes)
        assert len(sequence_proposals) == len(pred_bboxes)

        for i in range(len(sequence_proposals)):
            frame_gt_bboxes = np.asarray(gt_bboxes[i])
            frame_pred_bboxes = np.asarray(pred_bboxes[i])

            iou_thresh_metrics = []
            for iou_thresh in iou_thresholds:
                # IoU matching threshold
                precision, recall = match_bboxes_and_calc_precision_and_recall(frame_gt_bboxes, frame_pred_bboxes, iou_thresh=iou_thresh)
                #print(precision, recall)
                iou_thresh_metrics.append([precision, recall])

            frame_metrics.append(iou_thresh_metrics)
        #print(np.array(frame_metrics).shape)
        metrics.append(np.mean(frame_metrics, axis=0))

    metrics = np.asarray(metrics)
    # 'metrics' has shape (len(discrimination_thresholds), len(iou_thresholds), 2)
    metrics = np.swapaxes(metrics, 0, 1)
    # 'metrics' has shape (len(iou_thresholds), len(discrimination_thresholds), 2)
    metrics_dict = {iou_thresholds[i]: metrics[i] for i in range(metrics.shape[0])}
    
    return metrics_dict


