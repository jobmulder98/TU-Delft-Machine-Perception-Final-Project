import k3d
import numpy as np


def bgr_to_int(bgr_colors):
    """
    bgr_colors: np.array of size Nx3
    return: np.array of size N
    """
    return rgb_to_int(bgr_colors[:, ::-1])


def rgb_to_int(rgb_colors):
    """
    rgb_colors: np.array of size Nx3
    return: np.array of size N
    """
    # Avoid overflow issues with uint8.
    rgb_colors = rgb_colors.astype(np.uint32).T
    int_colors = (rgb_colors[0] << 16) + (rgb_colors[1] << 8) + rgb_colors[2]
    return int_colors


def get_default_camera(pose_transform=np.eye(4, 4)):
    # Homogenous camera positions.
    camera_pos = [0, 0, 0, 1]
    camera_focus_point = [10, 0, 0, 1]
    camera_up = [0, 0, 1, 1]
    default_camera = np.array([camera_pos, camera_focus_point, camera_up])

    pose_camera = pose_transform.dot(default_camera.T).T
    pose_camera_up = pose_camera[2, :3] - pose_camera[0, :3]
    return pose_camera[:2, :3].flatten().tolist() + pose_camera_up.tolist()


def plot_axes(T_plotorigin_target=np.eye(4, dtype=np.float32), length=1.0):
    """
    Creates k3d axes representation with red, green, blue vectors representing the
    x, y, z axis of the target frame within the plot's origin frame.

    T_plotorigin_target: homogeneous matrix
    length: length of the vectors

    returns: k3d.vectors object representing the axes of the target frame.
    """
    unit_vectors = np.asarray([[length, 0, 0, 1], [0, length, 0, 1], [0, 0, length, 1]], dtype=np.float32)
    start = [T_plotorigin_target[:3, 3]] * 3
    # Apply transform from right side.
    end = T_plotorigin_target.dot(unit_vectors.T).T[:, :3]
    pose_axes = k3d.vectors(
        origins=start,
        vectors=end[:, :3] - start,
        colors=[0xFF0000, 0xFF0000, 0x00FF00, 0x00FF00, 0x0000FF, 0x0000FF],  # 3x origin, head
    )
    return pose_axes



def plot_box(
    plot, label_camera, T_origin_camera=np.eye(4, dtype=np.float32), width=0.01, color=None, is_use_mesh=True, is_plot_axes=True
):
    """
    Add 3D box representation of `label_camera` to k3d plot `plot`.

    plot: k3d plot to visualize the labels in
    label_camera: label dictionary representing label in camera frame (see Measurements.get_labels_camera()).
    T_origin_camera: homogeneous transform transforming the labels given in camera frame to `origin`
        which represents the origin of the k3d plot.
    width: width of line representation (if is_use_mesh==False)
    color: color for all labels. If None: use per-class color
    is_use_meth: whether to use faster representation as k3d mesh (instead of lines)

    return: None
    """

    object_class_colors = {
        "Car": 0xFF0000,
        "Pedestrian": 0x00FF00,
        "Cyclist": 0x0000FF,
        "DontCare": 0xAAAAAA,
    }

    if color is None:
        object_class = label_camera["label_class"]
        object_class_color = object_class_colors[object_class]
    else:
        assert color <= 0xFFFFFF
        object_class_color = color

    try:
        from ipynb.fs.defs.practicum1 import get_corners_object
    except NameError as e:
        assert len(e.args) == 1
        e.args = (
            "Error when trying to import get_corners_object() from practicum 1 jupyter notebook.\n"
            "It might have failed due to requirements the ipynb.fs.defs module has on the notebook content,\n"
            "such as trying to import all-uppercase-letter variables (such as T, P2, ...).\n"
            "Please make sure to have variable names in lowercase.\n"
            "Here's the original error message:\n    " + e.args[0],
        )
        raise e

    # get corners in object frame
    corners_object = get_corners_object(label_camera["extent_object"])

    # transform corners to origin frame
    T_cam_object = label_camera["T_cam_object"]
    T_origin_object = T_origin_camera.dot(T_cam_object)
    corners_origin = T_origin_object.dot(corners_object.T).T

    if is_use_mesh:
        # opaque box surface
        BOX_MESH_INDICES = np.asarray([
            [0, 1, 2],  # bottom
            [0, 2, 3],  # bottom
            [0, 1, 5],  # front
            [0, 5, 4],  # front
            [2, 3, 7],  # back
            [2, 7, 6],  # back
            [1, 2, 6],  # right
            [1, 6, 5],  # right
            [0, 3, 7],  # left
            [0, 7, 4],  # left
            [4, 5, 6],  # top
            [4, 6, 7],  # top
            [0, 1, 5],  # front twice to be darker when opaque
            [0, 5, 4],  # front
        ], dtype=np.uint32)
        plot += k3d.mesh(vertices=corners_origin[:, 0:3], indices=BOX_MESH_INDICES, color=object_class_color, opacity=0.25)
    else:
        # draw lines as frame
        # index lists for drawing
        corner_indices_bottom = [0, 1, 2, 3, 0]
        corner_indices_top = [4, 5, 6, 7, 4]
        corner_indices_front_left = [0, 4]
        corner_indices_front_right = [1, 5]
        corner_indices_rear_left = [3, 7]
        corner_indices_rear_right = [2, 6]

        # draw box
        # bottom 4
        plot += k3d.line(corners_origin[corner_indices_bottom, 0:3], color=object_class_color, width=width)
        # top 4
        plot += k3d.line(corners_origin[corner_indices_top, 0:3], color=object_class_color, width=width)
        # vertical
        plot += k3d.line(corners_origin[corner_indices_front_left, 0:3], color=object_class_color, width=width)
        plot += k3d.line(corners_origin[corner_indices_front_right, 0:3], color=object_class_color, width=width)
        plot += k3d.line(corners_origin[corner_indices_rear_left, 0:3], color=object_class_color, width=width)
        plot += k3d.line(corners_origin[corner_indices_rear_right, 0:3], color=object_class_color, width=width)

    # draw axis
    if is_plot_axes:
        plot += plot_axes(T_origin_object)
