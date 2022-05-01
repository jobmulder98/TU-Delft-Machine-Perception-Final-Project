
import numpy as np
from sklearn.metrics import auc


def IOU(bboxa, bboxb):
    v1_a = bboxa[0] 
    u1_a = bboxa[1] 
    v2_a = bboxa[2]
    u2_a = bboxa[3]

    v1_b = bboxb[0] 
    u1_b = bboxb[1] 
    v2_b = bboxb[2]
    u2_b = bboxb[3]

    assert v1_a <= v2_a
    assert u1_a <= u2_a
    assert v1_b <= v2_b
    assert u1_b <= u2_b

    area1 = (v2_a-v1_a)*(u2_a-u1_a)
    area2 = (v2_b-v1_b)*(u2_b-u1_b)

    assert area1 >= 0.0
    assert area2 >= 0.0

    # Now we need to find the intersection box
    # to do that, find the largest (v, u) coordinates 
    # for the start of the intersection bounding box and 
    # the smallest (v, u) coordinates for the 
    # end of the intersection bounding box
    vv1 = max(v1_a, v1_b)
    uu1 = max(u1_a, u1_b)
    vv2 = min(v2_a, v2_b)
    uu2 = min(u2_a, u2_b)

    # So the intersection Bbox has the coordinates (vv1,uu1) (vv2,uu2)
    # compute the width and height of the intersection bounding box
    h = max(0, vv2 - vv1)
    w = max(0, uu2 - uu1)
    assert w >= 0.0
    assert h >= 0.0

    # find the intersection area
    intersection_area = w*h

    # find the union area of both the boxes
    union_area = area1 + area2 - intersection_area

    # compute the ratio of overlap between the computed
    # bounding box and the bounding box in the area list
    IOU = intersection_area / union_area

    assert IOU <= 1.
    assert IOU >= 0.

    return IOU


def mAP(metrics_dict):
    APs = []
    for _, metric_values in metrics_dict.items():
        y,x = metric_values.T
        try:
            auc_value = auc(x,y)
            APs.append(auc_value)
        except ValueError:
            print('x not monotonically decreasing. Ignoring.')

    mAP = np.mean(APs)
    return mAP


if __name__ == '__main__':
    b1 = [0, 0, 1, 1]
    b2 = b1
    #b2 = [2, 2, 3, 3]

    iou = IOU(b1, b2)
    print(iou)

