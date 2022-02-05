import numpy as np
import pybullet


def get_matrix_from_quaternion(orientation: np.ndarray) -> np.ndarray:
    assert orientation.shape == (4,), f"Invalid shape of orientation: {orientation.shape}"
    return np.array(pybullet.getMatrixFromQuaternion(orientation)).reshape((3, 3))
