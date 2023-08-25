from typing import Optional, List, Dict, Set, Type, Tuple, Union
import numpy as np
import numpy.typing as npt

import os
import pickle
from tqdm import tqdm

from datasets.ref_sampler.BaseRefSampler import BaseRefSampler
from datasets.BaseDataLoader import BaseDataLoader
from datasets.struct.data import Group, Sample


class MotionRefSampler(BaseRefSampler):
    """ Samples reference from past and or future for which d(tgt, ref) in [d_min, d_max]. If no such ref.-views exist,
        the sample is omitted.
    """
    motion_graph = {}

    def __init__(
        self,
        context: Optional[List[int]] = None,
        cache_graph: Optional[bool] = True,
        load_cached_graph: Optional[bool] = True,
        d_min: float = 0.3,
        d_max: float = 5,
        max_dist: float = 30,
        random_choice: bool = True,
        **kwargs,
    ):
        """
        Args:
            context: List of timesteps around t used as context - default = []
            cache_graph: True - distance between frames will be cached in pickle file
            load_cached_graph: True - distance between frames will be loaded from pickle file, False - calc from scratch
            d_min: Minimum distance between two successive frames (in meters)
            d_max: Maximum distance between two successive frames (in meters)
            max_dist: Maximum distance between any two frames
            random_choice: True - frames within [d_min, d_max] will be sampled randomly, False - sample closest frame
        """
        super(MotionRefSampler, self).__init__(**kwargs)
        self.context = context if context is not None else []
        self.graph = None
        self.cache_graph = cache_graph
        self.load_cached_graph = load_cached_graph
        self.d_min = d_min
        self.d_max = d_max
        self.max_dist = max_dist
        self.random_choice = random_choice

        self.rng = np.random.default_rng(2022)  # For consistent sampling across gpus and rep. results

        self.bwd_contexts = [ctx for ctx in context if ctx < 0]
        self.fwd_contexts = [ctx for ctx in context if ctx > 0]

        self.bwd_context = 0 if len(context) == 0 else - min(0, min(context))
        self.fwd_context = 0 if len(context) == 0 else max(0, max(context))

        self.fwd_context = [ctx for ctx in range(0, self.fwd_context + 1, +1) if ctx > 0]
        self.bwd_context = [ctx for ctx in range(0, -self.bwd_context - 1, -1) if ctx < 0]

    def initialize(
            self,
            dataset: Dict[str, List[Group]],
            dataset_path: str,
            cameras: Union[List[str], List[int]],
            dataloader: Type[BaseDataLoader],
            split: Optional[Union[List[str], Dict[str, List[int]]]] = None,
    ):
        # Get motion graph either from 1. Local cache, 2. Saved file cache, 3. Build from scratch
        cache_file = os.path.join(
            dataset_path,
            f'cached_motion_graph_{self.__class__.__name__}_{dataloader.__name__}.pickle'
        )
        if cache_file in MotionRefSampler.motion_graph:
            self.graph = MotionRefSampler.motion_graph[cache_file]
        else:
            if self.load_cached_graph and os.path.isfile(cache_file):
                with open(cache_file, 'rb') as handle:
                    self.graph = pickle.load(handle)
            else:
                self.graph = dict()
                for scene, groups in tqdm(dataset.items(), desc='Build scene graphs'):
                    self.graph[scene] = self._build_frame_graph(
                        groups=groups,
                        camera=cameras[0]
                    )
                if self.cache_graph:
                    with open(cache_file, 'wb') as handle:
                        pickle.dump(self.graph, handle, protocol=pickle.HIGHEST_PROTOCOL)
            MotionRefSampler.motion_graph[cache_file] = self.graph

        # create sample list
        for scene, groups in dataset.items():
            if (split is not None) and (scene not in split):
                continue
            for target_idx in range(len(groups)):
                if (split is not None) and (type(split) is dict) and (target_idx not in split[scene]):
                    continue

                fwd_idcs = self.sample_n(scene=scene, target_idx=target_idx, n=len(self.fwd_context), mode='fwd')
                bwd_idcs = self.sample_n(scene=scene, target_idx=target_idx, n=len(self.bwd_context), mode='bwd')
                idcs = list(bwd_idcs | fwd_idcs)
                idcs.sort()
                if len(idcs) == len(self.fwd_context) + len(self.bwd_context) + 1:
                    sample: Sample = {
                        'sequence': scene,
                        'groups': {i-len(self.bwd_context): dataset[scene][j] for i, j in enumerate(idcs)}
                    }
                    self.sample_list.append(sample)

    def sample_n(self, scene: Union[int, str], target_idx: int, n: int, mode: Optional[str] = 'fwd') -> Set[int]:
        """ Samples n reference views either from past or future (e.g. n=2 & mode='fwd' => t+1, t+2
        Args:
            scene: The current scene
            target_idx: Index of target frame
            n: The number of reference views which should be sampled
            mode: Which direction should be sampled: 'fwd', 'bwd'
        """
        assert mode in ['fwd', 'bwd'], f"Expect 'mode' to be in ['fwd', 'bwd'] but got {mode}"
        idx = target_idx
        idcs = [idx]
        while (len(idcs) < n + 1) and (idx in self.graph[scene]):
            # get other frames within distance window
            valid = (self.graph[scene][idx][1] > self.d_min) & (self.graph[scene][idx][1] < self.d_max)
            frames = self.graph[scene][idx][0][valid]
            new_idx = idx
            if mode == 'fwd':
                if np.count_nonzero(frames[frames > idx]):
                    if self.random_choice:
                        new_idx = self.rng.choice(frames[frames > idx])
                    else:
                        new_idx = frames[frames > idx][0]
            if mode == 'bwd':
                if np.count_nonzero(frames[frames < idx]):
                    if self.random_choice:
                        new_idx = self.rng.choice(frames[frames < idx])
                    else:
                        new_idx = frames[frames < idx][-1]
            idcs += [new_idx]
        idcs = set(idcs)
        return idcs

    def _build_frame_graph(
            self,
            groups: List[Group],
            camera: Union[str, int]
    ) -> Dict[int, Tuple[npt.NDArray[int], npt.NDArray[float]]]:
        """
        Args:
            groups: List of groups
            camera: Camera for which the distances between frames should be calculated
        Returns:
            graph: Dict of the form {i: ([j], [d(i, j)])} where [d(i, j)] is a list of distance
                   in m between frames I_i and I_j
        """
        graph = {}
        poses = np.stack([group['frames'][camera]['pose'] for group in groups])
        for i in range(poses.shape[0]):
            dist = np.linalg.norm(poses[:, 0:3, 3] - poses[i, 0:3, 3], 2, axis=1)
            j, = np.where(dist < self.max_dist)
            graph[i] = (j, dist[j])
        return graph
