from abc import ABC

import numpy as np

from typing import Dict, List
import os
import pickle

from datasets.struct.data import Group
from datasets.struct.r3d3_data import R3D3Data
from datasets.DDADDataset import DDADDataset
from datasets.utils.utils import get_others_r3d3
from vidar.utils.decorators import iterate1
from tqdm import tqdm


class DDADR3D3Dataset(DDADDataset, ABC):
    """ DDAD dataset with R3D3 generated poses, depth and confidence maps in 'others' field.
    Requires the following dataset structure:
    DDAD
    ├ $SCENE_DIR
        ├ ...
        └ r3d3
            └ $CAM_DIR
                └ xxxxxxxxxxxxxxxxx.npz
    ├ ...
    └ r3d3_poses.json
    Where:
        xxxxxxxxxxxxxxxxx.npz: Contains the following arrays
            'disp' - Geometrically inferred disparity of shape [1, H/8, W/8]
            'disp_up' - Upsampled geometrically inferred disparity of shape [1, H, W]
            'conf' - Depth confidence of shape [1, H/8, W/8] with values in range [0, 1]
        r3d3_poses.json: Nested Dict of the form {scene: {cam: {timestamp: [ego2world pose]}}} where ego2world
        is of shape [7] with [x, y, z, quaternion]
    """
    def __init__(self, **kwargs):
        super(DDADR3D3Dataset, self).__init__(**kwargs)

    def _check_labels(self, labels: List[str]):
        provided_labels = ['rgb', 'depth', 'pose', 'instance', 'mask', 'others_pose', 'others_disp_conf']
        if not all([label in provided_labels for label in labels]):
            raise ValueError("Only labels {} are provided by the DDAD dataset.".format(provided_labels))

    def _get_dataset(self) -> Dict[str, List[Group]]:
        """ Appends R3D3 data to 'others' field of DDADDataset
        """
        dataset_cache = os.path.join(self.path, f'cached_{DDADDataset.__name__}.pickle')
        if os.path.isfile(dataset_cache):
            with open(dataset_cache, 'rb') as pickle_file:
                dataset = pickle.load(pickle_file)
        else:
            dataset = super(DDADR3D3Dataset, self)._get_dataset()
        for scene in tqdm(dataset, desc='Scene'):
            for group in dataset[scene]:
                for cam in group['frames']:
                    data = {'pose': None, 'maps': None}
                    droid_file = group['frames'][cam]['rgb'].replace('rgb', 'r3d3').replace('.png', '.npz')
                    droid_file_abs = os.path.join(self.path, droid_file)
                    if os.path.isfile(droid_file_abs):
                        data['maps'] = droid_file
                        droid_data = np.load(droid_file_abs)
                        data['pose'] = droid_data['pose']
                    group['frames'][cam].update({'others': data})
        return dataset

    @staticmethod
    @iterate1
    def get_others(root: str, others: R3D3Data, labels: List[str]) -> Dict[str, np.ndarray]:
        return get_others_r3d3(root=root, others=others, labels=labels, default_shape=(1, 384, 640))
