

import scipy
import numpy as np
from metrics import IOU


class BoundingBox:
    def __init__(self, v, u, h, w, from_corners=False):
        # (v,u) defines the top corner of the bounding box
        assert isinstance(v, (int, np.integer)), type(v)
        assert isinstance(u, (int, np.integer)), type(u)
        assert isinstance(h, (int, np.integer)), type(h)
        assert isinstance(w, (int, np.integer)), type(w)

        self.v = v
        self.u = u
        if from_corners:
            self.h = h - v
            self.w = w - u
        else:
            self.h = h
            self.w = w

        assert self.w > 0
        assert self.h > 0

    def get_bbox_borders(self):
        return self.v, self.u, self.h, self.w

    def get_bbox_corners(self):
        return self.v, self.u, self.v + self.h, self.u + self.w 

    def get_bbox_corners_vis(self):
        return self.u, self.v, self.u + self.w, self.v + self.h 

    def is_within(self, point_uv):
        u, v = point_uv
        vmin, umin, vmax, umax = self.get_bbox_corners()
        return u >= umin and u < umax and v >= vmin and v < vmax

    def __repr__(self):
        return "BoundingBox(v=%d, u=%d, h=%d, w=%d)" % (self.v, self.u, self.h, self.w)

    def __eq__(self, other):
        return all(getattr(self, member) == getattr(other, member) for member in ('u', 'v', 'w', 'h'))


def match_bboxes(bbox_gt, bbox_pred, IOU_THRESH=0.0):
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
            iou_matrix[i, j] = IOU(bbox_gt[i,:], bbox_pred[j,:])

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


def clip_bbox_to_image(bbox, image_dims):
    v1, u1, v2, u2 = bbox.get_bbox_corners()
    v1 = np.clip(v1, 0, image_dims[0])
    u1 = np.clip(u1, 0, image_dims[1])
    v2 = np.clip(v2, 0, image_dims[0])
    u2 = np.clip(u2, 0, image_dims[1])
    
    h = v2 - v1
    w = u2 - u1

    # @TODO: find proper way of handling these cases
    if h <= 0:
        h = 1
    if w <= 0:
        w = 1

    bbox.v = v1
    bbox.u = u1
    bbox.h = h
    bbox.w = w

    

