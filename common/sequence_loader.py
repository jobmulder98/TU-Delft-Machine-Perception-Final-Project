import csv
import glob
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np


def load_sequences_info(sequences_csv_path):
    """
    Load the index range information from the sequences.csv file. Note that the
    range indices will be corrected, such that counting starts at zero.
    return: A dictionary that contains the start and end index inclusive
    (value) for each index (key).
    """
    info = {}
    with open(sequences_csv_path, newline="") as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        for line in reader:
            info[int(line[0])] = [int(line[1]) - 1, int(line[2]) - 1]
    return info


# rosradar and (kitti) radar frames are rotated by 180deg around x axis
# tf.euler_matrix(np.deg2rad(180), 0, 0)
T_rosradar_radar = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32)
T_radar_rosradar = np.linalg.inv(T_rosradar_radar)

# to get from roslidar to (kitti) lidar, we rotate by 90deg around the z axis
# tf.euler_matrix(0, 0, np.deg2rad(90))
T_roslidar_lidar = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
T_lidar_roslidar = np.linalg.inv(T_roslidar_lidar)


class SequenceLoader:
    def __init__(self, data_folder=os.environ["PRACTICUM3MP_DATA_DIR"], data_types=None, reference_time=0):
        self.data_folder = data_folder

        # Sort file list according to sample_idx_pattern.
        self.sample_idx_pattern = r"\d+"  # Old pattern: r'\d+\.\d+'
        self.sample_idx_re = re.compile(self.sample_idx_pattern)

        self.modality_specs = {
            "left": {
                "dir": os.path.normpath("lidar/training/image_2"),
                "loader": self.load_left_image,
            },
            "right": {
                "dir": os.path.normpath("lidar/training/image_3"),
                "loader": self.load_right_image,
            },
            "disparity": {
                "dir": os.path.normpath("lidar/training/disp"),
                "loader": self.load_disparity,
            },
            "lidar": {
                "dir": os.path.normpath("lidar/training/velodyne"),
                "calib_dir": os.path.normpath("lidar/training/calib"),
                "loader": self.load_point_cloud,
            },
            "radar": {
                "dir": os.path.normpath("radar/training/velodyne"),
                "calib_dir": os.path.normpath("radar/training/calib"),
                "loader": self.load_radar,
            },
            "pose": {
                "dir": os.path.normpath("kitti_summer_best_ones/training/pose"),
                "calib_dir": os.path.normpath("radar/training/calib"),
                "loader": self.load_pose,
            },
            "label": {
                "dir": os.path.normpath("lidar/training/label_2"),
                "loader": self.load_label,
            },
            #'disparity' : 'disp_img/disp_',  # Not available yet.
        }
        if data_types is None:
            data_types = self.modality_specs.keys()

        glob_strings = [self.modality_specs[data_type]["dir"] for data_type in data_types]

        file_list = sum([list(Path(data_folder).glob("{}/*".format(g))) for g in glob_strings], [])
        file_list = [str(f) for f in file_list]

        self.file_dict = {}
        for fname in file_list:
            idx = self.extract_sample_idx_from_filename(fname)
            if not idx in self.file_dict:
                self.file_dict[idx] = set()
            self.file_dict[idx].add(fname)

    def extract_sample_idx_from_filename(self, fname):
        bname = os.path.basename(fname)
        match = self.sample_idx_re.search(bname)
        if match is None:
            raise ValueError(
                "Sample index pattern {} did not match "
                "file {} with basename {}.".format(self.sample_idx_pattern, fname, bname)
            )
        return int(match.group(0))

    def load_left_image(self, fname, key="left"):
        left_img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        return {key: left_img}

    def load_right_image(self, fname, key="right"):
        right_img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        return {key: right_img}

    def load_disparity(self, fname, key="disparity"):
        disparity = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        return {key: disparity}

    def load_calibration(self, fname):
        # Copied from:
        # https://gitlab.tudelft.nl/eaipool/OpenPCDet/-/blob/master/pcdet/local/kitti_vis.py
        with open(fname, "r") as f:
            lines = f.readlines()
            P2 = np.array(lines[2].strip().split(" ")[1:], dtype=np.float32).reshape(3, 4)
            P3 = np.array(lines[3].strip().split(" ")[1:], dtype=np.float32).reshape(3, 4)
            V2C = np.array(lines[5].strip().split(" ")[1:], dtype=np.float32).reshape(3, 4)
            # make homogeneous
            V2C = np.vstack([V2C, np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)])

        return {"P2": P2, "P3": P3, "to_camera": V2C}

    def load_pose(self, fname, key="pose"):
        with open(fname, "r") as f:
            json_data = json.load(f)

        # Pose is given from rosradar to world frame (see static transforms above).
        # We give the pose with respect to the (kitti) radar frame.
        # Actually, giving it in the (kitti) lidar frame would be convenient, but we'd need to
        # either load it from a file of a sequence or hard code it here.
        T_world_rosradar = np.array(json_data["worldToLidar"], dtype=np.float32).reshape(4, 4)
        T_world_radar = T_world_rosradar.dot(T_rosradar_radar)
        return {key: T_world_radar}

    def load_label(self, fname, key="label"):
        with open(fname, "r") as f:
            labels = f.readlines()
        return {key: labels}

    def load_radar(self, fname, key="radar"):
        # Copied from:
        # https://gitlab.tudelft.nl/eaipool/OpenPCDet/-/blob/master/pcdet/local/kitti_vis.py
        scan = np.fromfile(fname, dtype=np.float32).reshape(-1, 11)
        scan = scan[scan[:, 2] > -0.5]  ## filter to remove points underground
        scan = scan[scan[:, 2] < 3.0]  ## filter to remove points above 8 ft
        scan = scan[scan[:, 0] > 3]  ## filter to remove points very close to vehicle

        return {key: scan}

    def load_point_cloud(self, fname, key="lidar"):
        point_cloud = np.fromfile(fname, dtype=np.float32).reshape(-1, 4)
        return {key: point_cloud}

    def get_file(self, fname):
        result = None
        sample_idx = self.extract_sample_idx_from_filename(fname)
        for modality, specs in self.modality_specs.items():
            if specs["dir"] in fname:
                # TODO: Either create directory in loader functions or here,
                # but doing it differently for the data and the calibration is
                # inconsistent.
                result = specs["loader"](fname, key=modality)
                if "calib_dir" in specs.keys():
                    calib_fname = os.path.join(self.data_folder, specs["calib_dir"], f"{sample_idx:05}.txt")
                    calibration = self.load_calibration(calib_fname)
                    result.update({f"{modality}_calibration": calibration})

        return sample_idx, result

    def get_first_index(self):
        return min(self.file_dict.keys())

    def __iter__(self):
        for index in sorted(self.file_dict.keys()):
            result = self[index]
            yield result

    def __len__(self):
        return len(self.file_dict.keys())

    def __getitem__(self, idx):
        current_idx = None
        collated_result = {}

        if not idx in self.file_dict:
            raise ValueError(f"Index {idx} is not in file_dict.")

        for fname in self.file_dict[idx]:
            sample_idx, result = self.get_file(fname)
            if sample_idx != idx:
                raise ValueError("Index mismatch between f{idx} and f{sample_idx}.")
            if result is None:
                print("No appropriate loader found for {}. Skipping.".format(fname))
                continue

            collated_result.update(result)

        if not collated_result:
            raise ValueError(f"Could not load any data for index {idx}. " f"Found files: {self.file_dict[idx]}.")

        collated_result.update({"sample_idx": current_idx})

        return collated_result


class LazySequenceLoaderDict:
    """
    Surrogates a result dict from the SequenceLoader, but loads data on demand.
    
    Avoids loading of all the data for a sample_idx, if only a small datum is needed (e.g. calib)
    Is roughly 2 orders of magnitude faster when only accessing calib.
    """
    def __init__(self, sequence_loader, sample_idx):
        if not sample_idx in sequence_loader.file_dict:
            raise ValueError(f"Index {sample_idx} is not in sequence_loader file_dict.")
        self.sequence_loader = sequence_loader
        self.sample_idx = sample_idx
        self.result = dict()  # cache
        self.allowed_modalities = {'left', 'right', 'disparity', 'lidar', 'lidar_calibration', 'radar', 'radar_calibration', 'label', 'pose', 'pose_calibration', 'sample_idx'}

    def __getitem__(self, modality):
        assert modality in self.allowed_modalities, modality

        # not in cache: load and update
        if modality not in self.result:
            # check specs (dir, calib_dir) for 'raw' (non-calib datum)
            modality_stripped = modality.replace("_calibration", "")  # strip _calibration suffix
            specs = self.sequence_loader.modality_specs[modality_stripped]
            is_calibration = modality.endswith("_calibration")
            
            if not is_calibration:
                # raw datum
                file_candidates = self.sequence_loader.file_dict[self.sample_idx]
                files = [f for f in file_candidates if specs["dir"] in f]
                if len(files) != 1:
                    raise KeyError(modality)
                fname = files[0]
                self.result.update(specs["loader"](fname, key=modality))  # update local cache
            else:
                # calib data of raw datum
                calib_fname = os.path.join(self.sequence_loader.data_folder, specs["calib_dir"], f"{self.sample_idx:05}.txt")
                calibration = self.sequence_loader.load_calibration(calib_fname)
                self.result.update({modality: calibration})
        
        # return result from cache
        return self.result[modality]

    def __contains__(self, modality):
        # try loading on demand, ignore keyerror, check cached result
        try:
            self.__getitem__(modality)
        except KeyError:
            pass
        return modality in self.result 
        

    def keys(self):
        # help users to find out which modalities are usable
        return self.allowed_modalities


class Measurements:
    """
    Encapsulates all sensor measurements of a given time step (index).
    """

    def __init__(self, sequence_loader, index, track_ids):
        self.index = index

        # whether to use LazySequenceLoader or load all data by default
        is_lazy = True
        if is_lazy:
            self.data_dict = LazySequenceLoaderDict(sequence_loader, index)
        else:
            self.data_dict = sequence_loader[index]
        self.track_ids = track_ids  # for current index, maps from label index to track_id
        self.has_used_iv_only_methods = False

    def get_index(self):
        return self.index

    def get_lidar_points(self):
        """
        Get lidar point cloud in lidar frame.
        N x ['x', 'y', 'z', 1]
        """
        pc_lidar = self.data_dict["lidar"]
        assert pc_lidar.shape[1] == 4
        pc_lidar[:, 3] = 1.0
        return pc_lidar

    def get_radar_measurements(self):
        """
        Get radar measurements in radar frame.
        N x ['x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'v_x', 'v_y', 'v_z', 'detectionconfidence', 'time']
        """
        return self.data_dict["radar"]

    def get_radar_points(self):
        """
        Get radar points in radar frame.
        N x ['x', 'y', 'z', 1]
        """
        radar_measurements = self.get_radar_measurements()
        radar_points = np.copy(radar_measurements[:, :4])
        radar_points[:, 3] = 1.0  # make homogeneous
        return radar_points

    def get_radar_compensated_radial_velocities(self):
        """
        Get radar compensated radial velocities in m/s for each radar point within radar frame.
        N
        """
        return self.get_radar_measurements()[:, 5]

    def get_camera_image(self):
        """H x W x [b,g,r]"""
        return self.data_dict["left"].copy()

    def get_right_camera_image(self):
        """H x W x [b,g,r]"""
        return self.data_dict["right"].copy()

    def get_disparity(self):
        """H x W"""
        return self.data_dict["disparity"].copy()

    def get_T_camera_lidar(self):
        return self.data_dict["lidar_calibration"]["to_camera"]

    def get_T_camera_radar(self):
        return self.data_dict["radar_calibration"]["to_camera"]

    def get_T_world_radar(self):
        """
        Pose from odometry.
        Represents the pose of the radar frame within a world-static frame
        """
        return self.data_dict["pose"]

    def get_camera_projection_matrix(self):
        return self.data_dict["lidar_calibration"]["P2"]

    def get_right_camera_projection_matrix(self):
        return self.data_dict["lidar_calibration"]["P3"]

    def get_labels_raw(self):
        return self.data_dict["label"]

    def get_label_dicts(self, is_verbose=True):

        labels_raw = self.get_labels_raw()
        label_dicts = []
        for label_index, act_line in enumerate(labels_raw):
            act_line = act_line.split()
            (
                label_class,
                _,
                occlusion_level,
                _,
                x1,
                y1,
                x2,
                y2,
                h_object,
                w_object,
                l_object,
                x_cam,
                y_cam,
                z_cam,
                yaw_lidar_rad,
                score,
            ) = act_line
            x1, y1, x2, y2, h_object, w_object, l_object, x_cam, y_cam, z_cam, yaw_lidar_rad, score = map(
                float, [x1, y1, x2, y2, h_object, w_object, l_object, x_cam, y_cam, z_cam, yaw_lidar_rad, score]
            )

            occlusion_level, x1, y1, x2, y2 = map(int, [occlusion_level, x1, y1, x2, y2])

            yaw_lidar_rad = -(yaw_lidar_rad + np.pi / 2)  ## undo changes made to rotation

            if is_verbose and label_class == "Pedestrian" and not label_index in self.track_ids:
                print("warning: track_id missing for pedestrian label")
                # assert False

            label_dict = {
                "label_class": label_class,
                "2d_bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "yaw_lidar_rad": yaw_lidar_rad,
                "bottom_center_cam": np.asarray([x_cam, y_cam, z_cam], dtype=np.float32),
                "occlusion_level": occlusion_level,
                "extent_object": np.asarray([l_object, w_object, h_object], dtype=np.float32),
                "score": score,
                "track_id": self.track_ids.get(label_index, None),
            }
            label_dicts.append(label_dict)

        return label_dicts

    def get_labels_camera(self, is_verbose=True):
        """
        Get labels in camera frame.
        """

        # TODO if below import is too slow, consider importing on module scope
        # i.e., make get_T_cam_object a module-wide function variable, initialize it
        # with None and set it on-demand (kinda singleton pattern).
        try:
            from ipynb.fs.defs.practicum1 import get_T_cam_object
        except NameError as e:
            assert len(e.args) == 1
            e.args = (
                "Error when trying to import get_T_cam_object() from practicum 1 jupyter notebook.\n"
                "It might have failed due to requirements the ipynb.fs.defs module has on the notebook content,\n"
                "such as trying to import all-uppercase-letter variables (such as T, P2, ...).\n"
                "Please make sure to have variable names in lowercase.\n"
                "Here's the original error message:\n    " + e.args[0],
            )
            raise e

        label_dicts = self.get_label_dicts(is_verbose=is_verbose)
        T_cam_lidar = self.get_T_camera_lidar()
        label_dicts_camera = []

        for label_dict in label_dicts:

            # assemble transform of object in camera frame
            T_cam_object = get_T_cam_object(label_dict, T_cam_lidar)
            label_dict_camera = dict(label_dict)  # copy to add T_cam_object inline
            label_dict_camera["T_cam_object"] = T_cam_object
            assert T_cam_object.dtype == np.float32, T_cam_object.dtype

            # remove obsolete intermediate representations.
            # yaw and bottom center are implicitly covered in T_cam_object
            del label_dict["yaw_lidar_rad"]
            del label_dict["bottom_center_cam"]

            label_dicts_camera.append(label_dict_camera)

        return label_dicts_camera

    def get_timestamp_ms(self):
        """
        Heuristics for now. Assume 10 Hz in between measurements.
        """
        return float(self.index) / 10.0

    def get_ground_plane(self):
        """
        IV only.

        Get ground plane in [a, b, c, d] format for current frame_index.
        The plane is defined by all points [x,y,z] which fulfill ax + by + cz + d = 0.
        The plane is defined in camera frame.

        Returns [a, b, c, d]  plane parameters
        """
        # mark 'dirty' flag, which MP students should not use
        # well - we're aware that there's workarounds, but we will check!
        # MP students: we know you can compute them yourselves - just check practicum 3 mp
        self.has_used_iv_only_methods = True
        iv_only_ground_planes_file = os.path.join(os.environ["SOURCE_DIR"], "assignment", "iv_only_ground_planes.json")
        if not os.path.exists(iv_only_ground_planes_file):
            print(
                ""
                "warning: ground planes file not found.\n"
                "  If you're an MP student, this is normal. Don't use this method and compute the ground planes yourself.\n"
                "  If you're an IV student, this should not happen. Contact the staff.\n"
                "  Has the file %s been shipped to you?\n" % iv_only_ground_planes_file
            )
            return None

        with open(iv_only_ground_planes_file) as fp:
            ground_planes = json.load(fp)
        # cast keys from string to int
        ground_planes = {int(frame_index): ground_plane for frame_index, ground_plane in ground_planes.items()}

        if self.get_index() not in ground_planes.keys():
            print(
                "warning: frame_index %d not found in precomputed ground planes. Are you using the proper frames within the sequence?"
                % self.get_index()
            )
            return None
        else:
            return np.asarray(ground_planes[self.get_index()])

    def get_T_newworld_camera(self):
        """
        Get transformation from camera frame into newworld frame.
        Newworld is a world-static frame, which has its origin at the radar frame during frame 1430.
        The newworld frame is oriented with
          x in driving direction of the vehice,
          y to the left,
          z to the top.


        returns None if file not present (MP students), or out-of-bounds access (IV students).
          T_newworld_cam for the current frame otherwise.
        """

        # mark 'dirty' flag, which MP students should not use
        # well - we're aware that there's workarounds, but we will check!
        # MP students: we know you can compute them yourselves.
        self.has_used_iv_only_methods = True

        iv_only_ts_newworld_cam_file = os.path.join(
            os.environ["SOURCE_DIR"], "assignment", "iv_only_ts_newworld_cam.json"
        )
        if not os.path.exists(iv_only_ts_newworld_cam_file):
            print(
                ""
                "warning: Ts_newworld_cam file not found.\n"
                "  If you're an MP student, this is normal. Don't use this method and compute the ground planes yourself.\n"
                "  If you're an IV student, this should not happen. Contact the staff.\n"
                "  Has the file %s been shipped to you?\n" % iv_only_ts_newworld_cam_file
            )
            return None

        with open(iv_only_ts_newworld_cam_file) as fp:
            Ts_newworld_cam = json.load(fp)
        # cast keys from string to int
        Ts_newworld_cam = {int(frame_index): ground_plane for frame_index, ground_plane in Ts_newworld_cam.items()}

        if self.get_index() not in Ts_newworld_cam.keys():
            print(
                "warning: frame_index %d not found in precomputed Ts_newworld_cam. Are you using the proper frames within the sequence?"
                % self.get_index()
            )
            return None
        else:
            return np.asarray(Ts_newworld_cam[self.get_index()], dtype=np.float32)


class Sequence:
    """
    Encapsulates all Measurements from consecutive time steps (indices) of a sequence index.
    """

    def __init__(self, sequence_index, start_index, end_index):
        self.sequence_index = sequence_index
        self.start_index = start_index
        self.end_index = end_index
        self.sequence_loader = SequenceLoader()

        self.track_ids = dict()
        self.__load_track_ids()

    def __load_track_ids(self):
        """
        Load track ids of pedestrian labels.
        """

        track_ids_file = os.path.join(os.environ["PRACTICUM3MP_DATA_DIR"], "track-ids.json")
        if os.path.exists(track_ids_file):
            with open(track_ids_file) as fp:
                self.track_ids = json.load(fp)

            self.track_ids = {int(k): v for k, v in self.track_ids.items()}  # make frame_ids ints
            for frame_index, label_track_ids in self.track_ids.items():  # make label_index ints
                self.track_ids[frame_index] = {int(k): v for k, v in label_track_ids.items()}
        else:
            print("warning: track_id file not existant")
            self.track_ids = dict()

        frames_with_track_ids = set(self.track_ids.keys())
        sequence_frames = set(range(self.start_index, self.end_index + 1))
        if not sequence_frames.issubset(frames_with_track_ids):
            print("warning: not all frames of current sequence have track_ids.")

    def get_indices(self):
        return range(self.start_index, self.end_index + 1)

    def get_measurements(self, index):
        """
        Get `Measurements` object for time-index `index`, which
        encapsulates the set of measurements of multiple sensors.
        """
        assert index >= self.start_index
        assert index <= self.end_index
        return Measurements(self.sequence_loader, index, self.track_ids.get(index, dict()))

    def __getitem__(self, index):
        return self.get_measurements(index)

    def __iter__(self):
        for index in range(self.start_index, self.end_index + 1):
            yield Measurements(self.sequence_loader, index, self.track_ids.get(index, dict()))

    def __len__(self):
        return self.end_index - self.start_index + 1


class Dataset:
    """
    Encapsulates Sequences.
    """

    def __init__(self, sequences_csv_path=os.path.join(os.environ["PRACTICUM3MP_DATA_DIR"], "sequences.csv")):
        self.sequence_info = load_sequences_info(sequences_csv_path)
        self.min_index = min([si[0] for si in self.sequence_info.values()])
        self.max_index = max([si[1] for si in self.sequence_info.values()])
        self.min_sequence_index = min(self.sequence_info.keys())
        self.max_sequence_index = max(self.sequence_info.keys())

    def get_sequence_indices(self):
        return self.sequence_info.keys()

    def get_sequence(self, sequence_index):
        start_index, end_index = self.sequence_info[sequence_index]
        return Sequence(sequence_index=sequence_index, start_index=start_index, end_index=end_index)

    def __getitem__(self, index):
        return self.get_sequence(index)

    def __iter__(self):
        for index in range(self.min_sequence_index, self.max_sequence_index + 1):
            yield self.get_sequence(index)

    def get_custom_sequence(self, start_index, end_index):
        """
        Create sequence from custom start_index and end_index (incl.).
        """
        assert start_index >= self.min_index
        assert end_index <= self.max_index
        return Sequence(sequence_index=None, start_index=start_index, end_index=end_index)
