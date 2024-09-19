"""
Various helper functions for the ZED
"""

import numpy as np


def to_intrinsic_matrix(camera_parameters):
    """
    =INPUT=
        camera_parameters - instance of CameraParameters class
            Holds the intrinsic calibration parameters
    =OUTPUT=
        intrinsics - ndarray of shape (3, 3)
            Intrinsic calibration as 3x3 matrix
    """

    intrinsics = np.zeros((3, 3), dtype='float')
    intrinsics[0, 0] = camera_parameters.fx
    intrinsics[0, 2] = camera_parameters.cx
    intrinsics[1, 1] = camera_parameters.fy
    intrinsics[1, 2] = camera_parameters.cy
    intrinsics[2, 2] = 1
    return intrinsics


def split_frame(side_by_side_frame):
    """
    Split a side-by-side numpy array frame into separate left and right arrays.
    """
    width = side_by_side_frame.shape[0]
    frame_left, frame_right = np.array_split(side_by_side_frame,2,1)

    return frame_left, frame_right