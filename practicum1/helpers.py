import numpy as np

def get_class_2d_bboxes_from_labels(camera_labels, label_class):
    filtered_labels = [camera_labels_dict['2d_bbox'] for camera_labels_dict in camera_labels if camera_labels_dict['label_class'] == label_class] 
    return filtered_labels
    

