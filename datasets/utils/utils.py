import os
import numpy as np
import numpy.typing as npt
import cv2

from datasets.struct.r3d3_data import R3D3Data
from typing import Tuple, Type, Optional, List, Union, Dict
from functools import partial
from multiprocessing import Pool, Lock, Manager
from datasets.struct.data import Dataset, Group
from datasets.BaseDataLoader import BaseDataLoader

from vidar.datasets.OuroborosDataset import generate_proj_maps
from dgp.utils.camera import Camera
from dgp.utils.pose import Pose


def intrinsics_4_to_nxn(intrinsics: npt.NDArray, dim: Optional[int] = 3) -> npt.NDArray:
    """ Converts intrinsics in 1d compact format to matrix
    Args:
        intrinsics: (4)
        dim: 3 / 4
    Returns:
        intrinsics_matrix: (dim, dim)
    """
    intrinsics_matrix = np.eye(dim, dtype=intrinsics.dtype)
    intrinsics_matrix[0, 0] = intrinsics[0]
    intrinsics_matrix[1, 1] = intrinsics[1]
    intrinsics_matrix[0, 2] = intrinsics[2]
    intrinsics_matrix[1, 2] = intrinsics[3]
    return intrinsics_matrix


def _check_depth_maps_groups(
        groups: List[Group],
        path: str,
        data_loader: Type[BaseDataLoader],
        shape: Tuple[int, int],
        lock: Lock
) -> None:
    for group in groups:
        if group['lidar'] is not None:
            if not all([os.path.isfile(os.path.join(path, frame['depth'])) for frame in group['frames'].values()]):
                points = data_loader.get_points(os.path.join(path, group['lidar']))
                lidar_pose = Pose.from_matrix(group['lidar_pose'])
                points_w = lidar_pose * points[:, :3]
                for cam, frame in group['frames'].items():
                    intrinsics = intrinsics_4_to_nxn(frame['intrinsics'], dim=3)
                    pose = Pose.from_matrix(frame['pose']).inverse()
                    camera = Camera(K=intrinsics, p_cw=pose)
                    depth_map = generate_proj_maps(camera, points_w, shape)[None]
                    depth_map_file = os.path.join(path, frame['depth'])
                    lock.acquire()
                    try:
                        if not os.path.exists(os.path.dirname(depth_map_file)):
                            os.makedirs(os.path.dirname(depth_map_file))
                    finally:
                        lock.release()
                    np.savez_compressed(depth_map_file, depth=depth_map)


def check_depth_maps(
        path: str,
        dataset: Dataset,
        data_loader: Type[BaseDataLoader],
        shape: Tuple[int, int],
        n_procs: Optional[int] = 1
) -> None:
    manager = Manager()
    lock = manager.Lock()
    with Pool(processes=n_procs) as pool:
        pool.map(
            partial(_check_depth_maps_groups, path=path, data_loader=data_loader, shape=shape, lock=lock),
            list(dataset.values())
        )


def get_others_r3d3(
        root: str,
        others: R3D3Data,
        labels: List[str],
        default_shape: Tuple[int, int, int]
) -> Dict[str, np.ndarray]:
    """ Returns others field for r3d3 data containing pose, depth and confidence
    Args:
        root: Path to dataset root
        others: Buffered others field containing pose and path to R3D3 maps
        labels: Labels to be loaded ('others_pose' and/or 'others_disp_conf')
        default_shape: Default R3D3 disparity or confidence shape of the form (C, H, W)
    Returns:
        Loaded others data as dict of numpy arrays
    """
    data = {}
    if 'others_pose' in labels:
        if others['pose'] is not None:
            pose = others['pose'].astype(np.float32)
        else:
            pose = np.zeros(7, dtype=np.float32)
        data.update({'pose': pose})
    if 'others_disp_conf' in labels:
        data.update(load_r3d3_disp_conf(
            filename=os.path.join(root, others['maps']) if others['maps'] is not None else None,
            default_shape=default_shape
        ))
    return data


def load_r3d3_disp_conf(
        filename: Union[str, None],
        default_shape: Tuple[int, int, int]
) -> Dict[str, npt.NDArray[np.float32]]:
    """ Loads R3D3 disparity and confidence maps
    Args:
        filename: Path to R3D3 training file
        default_shape: Default R3D3 disparity or confidence shape of the form (C, H, W)
    Returns:
        Dictionary of loaded disparities in full and 1/8th resolution. If the maps are not available (file does not
        exist), empty maps (all 0) are returned. The dictionary is of the form
        {'disp': (C, H/8, W/8), 'disp_up': (C, H, W), 'conf': (C, H/8, W/8), 'conf_up': (C, H, W)}
    """
    if filename is not None:
        npz_data = np.load(filename)
        disp = npz_data['disp']
        disp_up = npz_data['disp_up']
        conf = npz_data['conf']
        disp = disp.reshape(1, *disp.shape[-2:])
        disp_up = disp_up.reshape(1, *disp_up.shape[-2:])
        conf = conf.reshape(1, *conf.shape[-2:])
        _, h, w = disp_up.shape
        conf_up = cv2.resize(conf[0], dsize=(w, h), interpolation=cv2.INTER_LINEAR)[None]
    else:
        default_shape_down = (default_shape[0], default_shape[1] // 8, default_shape[2] // 8)
        disp = np.zeros(default_shape_down, dtype=np.float32)
        conf = np.zeros(default_shape_down, dtype=np.float32)
        disp_up = np.zeros(default_shape, dtype=np.float32)
        conf_up = np.zeros(default_shape, dtype=np.float32)
    return {
        'disp': disp,
        'disp_up': disp_up,
        'conf': conf,
        'conf_up': conf_up
    }

