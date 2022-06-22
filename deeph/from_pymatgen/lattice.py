import itertools
import numpy as np


# The following internal methods are used in the get_points_in_sphere method.
def _compute_cube_index(coords: np.ndarray, global_min: float, radius: float
                        ) -> np.ndarray:
    """
    Compute the cube index from coordinates
    Args:
        coords: (nx3 array) atom coordinates
        global_min: (float) lower boundary of coordinates
        radius: (float) cutoff radius

    Returns: (nx3 array) int indices

    """
    return np.array(np.floor((coords - global_min) / radius), dtype=int)

def _three_to_one(label3d: np.ndarray, ny: int, nz: int) -> np.ndarray:
    """
    The reverse of _one_to_three
    """
    return np.array(label3d[:, 0] * ny * nz +
                    label3d[:, 1] * nz + label3d[:, 2]).reshape((-1, 1))

def _one_to_three(label1d: np.ndarray, ny: int, nz: int) -> np.ndarray:
    """
    Convert a 1D index array to 3D index array

    Args:
        label1d: (array) 1D index array
        ny: (int) number of cells in y direction
        nz: (int) number of cells in z direction

    Returns: (nx3) int array of index

    """
    last = np.mod(label1d, nz)
    second = np.mod((label1d - last) / nz, ny)
    first = (label1d - last - second * nz) / (ny * nz)
    return np.concatenate([first, second, last], axis=1)

def find_neighbors(label: np.ndarray, nx: int, ny: int, nz: int):
    """
    Given a cube index, find the neighbor cube indices

    Args:
        label: (array) (n,) or (n x 3) indice array
        nx: (int) number of cells in y direction
        ny: (int) number of cells in y direction
        nz: (int) number of cells in z direction

    Returns: neighbor cell indices

    """

    array = [[-1, 0, 1]] * 3
    neighbor_vectors = np.array(list(itertools.product(*array)),
                                dtype=int)
    if np.shape(label)[1] == 1:
        label3d = _one_to_three(label, ny, nz)
    else:
        label3d = label
    all_labels = label3d[:, None, :] - neighbor_vectors[None, :, :]
    filtered_labels = []
    # filter out out-of-bound labels i.e., label < 0
    for labels in all_labels:
        ind = (labels[:, 0] < nx) * (labels[:, 1] < ny) * (labels[:, 2] < nz) * np.all(labels > -1e-5, axis=1)
        filtered_labels.append(labels[ind])
    return filtered_labels