import matplotlib.pyplot as plt
import cv2

# Common imports
import numpy as np
import os

from ImagePatch import ImagePatch
from BoundingBox import BoundingBox


def sliding_window(image, step_sizes, window_size):
    """
    step_sizes: [h, w]
    window_size: [h, w]

    returns: (v, u, window_size[0], window_size[1]) == (v, u, h, w)
    """
    assert len(step_sizes) == 2
    assert len(window_size) == 2

	# slide a window across the image
    y_min = 0
    y_max = 1 
    x_min = 0
    x_max = 1

    y_min = int(y_min * image.shape[0])
    y_max = int(y_max * image.shape[0])
    x_min = int(x_min * image.shape[1])
    x_max = int(x_max * image.shape[1])

    for y in range(y_min, y_max, step_sizes[0]):
	    for x in range(x_min, x_max, step_sizes[1]):
		    # yield the current window
		    yield (y, x, image[y:y + window_size[0], x:x + window_size[1]])


def resize_image(image, height=None, width=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    elif height is None:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    else:
        dim = (width, height)

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def generate_region_proposals(image, anchors, cheat=True, vis=False):
    if cheat and not vis:
        # We will cheat and only consider the bottom half of the image...
        rows_cut = image.shape[0]//2
        image = image[rows_cut:]

    region_proposals = []
    for anchor in anchors:
        if vis:
            step_sizes = [anchor_dim for anchor_dim in anchor]
        else:
            step_sizes = [anchor_dim//4 for anchor_dim in anchor]
        for proposal in sliding_window(image, step_sizes=step_sizes, window_size=anchor):
            proposal_patch = proposal_to_ImagePatch(proposal)
            if cheat and not vis:
                proposal_patch.bbox.v += rows_cut 
            region_proposals.append(proposal_patch)

    return region_proposals


def generate_sequence_proposals(sequence, anchors, **kwargs):
    sequence_proposals = []
    for image in sequence:
        region_proposals = generate_region_proposals(image, anchors, **kwargs)
        sequence_proposals.append(region_proposals)

    sequence_proposals = np.asarray(sequence_proposals)
    return sequence_proposals


def proposal_to_ImagePatch(proposal):
    y = proposal[0]
    x = proposal[1]
    patch = proposal[2]
    h, w = calculate_hw(patch)

    bbox = BoundingBox(y,x,h,w)
    return ImagePatch(patch, bbox)


def calculate_hw(bbox):
    h = bbox.shape[0]
    w = bbox.shape[1]

    return h,w 


