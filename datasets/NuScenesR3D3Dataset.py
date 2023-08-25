from abc import ABC

import numpy as np
from tqdm import tqdm

from typing import Dict, List
import os
import pickle

from datasets.struct.data import Group
from datasets.struct.r3d3_data import R3D3Data
from datasets.NuScenesDataset import NuScenesDataset
from datasets.utils.utils import get_others_r3d3
from vidar.utils.decorators import iterate1


class NuScenesR3D3Dataset(NuScenesDataset, ABC):
    def __init__(self, **kwargs):
        super(NuScenesR3D3Dataset, self).__init__(**kwargs)

    def _check_labels(self, labels: List[str]):
        provided_labels = ['rgb', 'depth', 'pose', 'lidar', 'boxes3d', 'mask', 'others_pose', 'others_disp_conf']
        if not all([label in provided_labels for label in labels]):
            raise ValueError("Only labels {} are provided by the NuScenes dataset.".format(provided_labels))

    def _get_dataset(self) -> Dict[str, List[Group]]:
        """
        Returns: Dictionary of data samples in self.path, i.e. {'scene': [Group]}
        """
        dataset_cache = os.path.join(self.path, f'cached_{NuScenesDataset.__name__}.pickle')
        if os.path.isfile(dataset_cache):
            with open(dataset_cache, 'rb') as pickle_file:
                dataset = pickle.load(pickle_file)
        else:
            dataset = super(NuScenesR3D3Dataset, self)._get_dataset()
        for scene in tqdm(dataset, desc='Scene'):
            for group in dataset[scene]:
                for cam in group['frames']:
                    data = {'pose': None, 'maps': None}
                    droid_file = group['frames'][cam]['rgb'].replace('CAM_', 'R3D3_').replace('.jpg', '.npz')
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
        return get_others_r3d3(root=root, others=others, labels=labels, default_shape=(1, 448, 768))
