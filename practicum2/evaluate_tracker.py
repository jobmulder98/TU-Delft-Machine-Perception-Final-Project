import sys
import os
import argparse
from multiprocessing import freeze_support
from collections import defaultdict
from Object import Object
import numpy as np

import trackeval

def objects_to_tracker(objects, bb_size=2):
    """
    Convert tracks into the format that the HOTA code wants. 

    Example
    
    <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    1,1,20,50,10,20,-1,-1,-1,-1
    1,2,30,40,10,20,-1,-1,-1,-1
    2,5,10,40,10,20,-1,-1,-1,-1
    3, ...
    ...

    will be converted to
    {
    '1': [['1','1','20','50','10','20','-1','-1','-1','-1'],
          ['1','2','30','40','10','20','-1','-1','-1','-1']],
    '2': [['2','5','10','40','10','20','-1','-1','-1','-1']],
    '3': [['3', ...]...],
     ...
     }

    <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    The conf value contains the detection confidence.
    For the ground truth, it acts as a flag whether the entry is to be considered. 
    A value of 0 means that this particular instance is ignored in the evaluation, 
    while any other value can be used to mark it as active. 
    The world coordinates x,y,z are ignored for the 2D challenge 
    and can be filled with -1.    
    """
    gt = defaultdict(list)

    for idx, obj in enumerate(objects, start=1):

        for frame, pos in zip(obj.ts, obj.pos.T):

            frame += 1

            assert frame > 0, 'no time-stamps < 0 allowed'

            gt[str(frame)].append([str(frame),
                                   str(idx),
                                   str(pos[0]-bb_size/2.),
                                   str(pos[1]+bb_size/2.),
                                   str(bb_size),
                                   str(bb_size),
                                   '-1','-1','-1','-1'])
    return gt

def kf_to_object(kf):
    pos = kf.mu_upds
    pos = [p[:2,0] for p in pos]
    pos = np.stack(pos).T

    kf_obj = Object(pos=pos, ts=kf.ts)

    return kf_obj

def kfs_to_objects_multi(kfs):
    """
    transforms kfs into Object
    """
    return list(map(kf_to_object, kfs))

def calculate_hota(gt_objs, kfs, bb_size=2., seq_length=None):
    """
    Calculates HOTA metric given ground-truth objects and tracking objects
    """
    assert all(map(lambda x: type(x).__name__.endswith("Object"), gt_objs))
    tracker_objs = kfs_to_objects_multi(kfs)

    return evaluate_tracker(gt_objs, tracker_objs, bb_size, seq_length)
    

# python3 -m pdb scripts/run_mot_challenge.py --BENCHMARK practicum2 --DO_PREPROC False
def evaluate_tracker(gt_objs, tracker_objs, bb_size=1, seq_length=None):
    """
    run_mot_challenge.py script adapted to our purpose. 
    Also some code changes to mot_challenge_2d_box.py (mostly prevent loading from files
    and replace with desired input).
    Right now, lots of HOTA code is unnecessary junk.

    ----------------------------------------------------------------------------
    How to use
    ----------------------------------------------------------------------------
    You have N ground-truth objects with their x-y positions in an 2xT numpy
    array and the corresponding T time-steps (use int's) in a list/ numpy array 
    of length T. time-steps should be >= 0.

    You have

    xy_list - list of length N of 2xT numpy arrays each corresponding to one object
    ts_list - list of length N of lists/numpy arrays of length T each 
              corresponding to one object

    xy_list[i] and ts_list[i] are the same object (i arbitrary)

    That's how you turn these into gt_objs.

    import Object from Object

    gt_objs = []
    for xy, ts in zip(xy_list, ts_list):
        gt_objs.append(Object(pos=xy, ts=ts)

    Same for tracker objects. 

    If you do not provide ts, Object will initializes ts as list(range(T))
    ----------------------------------------------------------------------------
    gt_objs     : ground-truth objects
    tracker_objs: tracker objects
    bb_size     : size of bounding box around gt and tracked objects, value is
                  arbitrary, does have an impact on HOTA
    seq_legth   : length of sequence. It is not entirely clear what this does, but
                  you probably want it to be at least as much as the length of your 
                  sequence (number of time-steps)
    returns     : dictionary with final evaluation metric
    """
    assert all(map(lambda x: type(x).__name__.endswith("Object"), gt_objs))
    assert all(map(lambda x: type(x).__name__.endswith("Object"), tracker_objs))

    if seq_length is None:
        # compute sequence length from given tracker/gt objects (max of ts)
        max_ts = max(max(tracker_obj.ts) for tracker_obj in tracker_objs)
        max_ts = max(max_ts, max(max(gt_obj.ts for gt_obj in gt_objs)))
        min_ts = min(min(tracker_obj.ts) for tracker_obj in tracker_objs)
        min_ts = min(min_ts, min(min(gt_obj.ts for gt_obj in gt_objs)))

        assert min_ts == 0, "min timestamp (of objects and GT) is not 0"
        seq_length = max_ts + 1

    gt = objects_to_tracker(gt_objs, bb_size)
    tracker = objects_to_tracker(tracker_objs, bb_size)

    freeze_support()

    # Command line interface:
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['DISPLAY_LESS_PROGRESS'] = False
    default_dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    default_metrics_config = {'METRICS': ['HOTA'], 'THRESHOLD': 0.5}
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs

    config['DO_PREPROC'] = False
    config['BENCHMARK'] = 'practicum2'

    config['OUTPUT_DETAILED'] = False
    config['OUTPUT_SUMMARY'] = False
    config['OUTPUT_EMPTY_CLASSES'] = False
    config['PLOT_CURVES'] = False
    config['TIME_PROGRESS'] = False
    config['LOG_ON_ERROR'] = None
    config['PRINT_CONFIG'] = False

    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}

    # Run code
    evaluator = trackeval.Evaluator(eval_config)

    # Here we provide the input that the HOTA code would usually load from files
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config, 
                                                         seq_lengths={'practicum2-01': seq_length},
                                                         seq_list=['practicum2-01'],
                                                         tracker_list=['tracker'],
                                                         read_gt=gt,
                                                         read_tracker=tracker)]

    metrics_list = []
    for metric in [trackeval.metrics.HOTA]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric(metrics_config))
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')

    evaluator.evaluate(dataset_list, metrics_list)
    res = evaluator.output_res['MotChallenge2DBox']['tracker']['practicum2-01']['pedestrian']
    return res
