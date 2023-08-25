from typing import Optional, Dict, List, Union, Tuple, Type

import numpy as np

from datasets.struct.data import Group
from datasets.struct.r3d3_data import R3D3Data
from datasets.ref_sampler.MotionRefSampler import MotionRefSampler
from datasets.BaseDataLoader import BaseDataLoader


class R3D3MotionRefSampler(MotionRefSampler):
    """ Samples reference views based on R3D3 generated pose predictions """
    def __init__(self, *args, **kwargs):
        super(R3D3MotionRefSampler, self).__init__(*args, **kwargs)
        self.dataset_path: Union[str, None] = None
        self.dataloader: Union[Type[BaseDataLoader], None] = None

    def initialize(self, dataset_path: str, dataloader: Type[BaseDataLoader], *args, **kwargs):
        self.dataset_path = dataset_path
        self.dataloader = dataloader
        super(R3D3MotionRefSampler, self).initialize(
            dataset_path=dataset_path,
            dataloader=dataloader,
            *args, **kwargs
        )

    def _build_frame_graph(
        self,
        groups: List[Group],
        camera: Union[str, int]
    ) -> Dict[int, Tuple[int, float]]:
        """ Build frame graph based on R3D3 generated poses instead of using gt poses """
        graph = {}
        others: List[R3D3Data] = [group['frames'][camera]['others'] for group in groups]
        valid = np.array([other['pose'] is not None for other in others])
        poses = np.stack([other['pose'] if other['pose'] is not None else np.zeros(7) for other in others])
        for i, pose in enumerate(poses):
            if valid[i]:
                dist = np.linalg.norm(poses[:, 0:3] - pose[0:3], 1, axis=1)
                j, = np.where((dist < self.max_dist) & valid)
                graph[i] = (j, dist[j])
        return graph

