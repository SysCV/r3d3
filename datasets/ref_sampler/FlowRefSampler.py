from typing import Optional, List, Dict, Callable, Type, Tuple, Union
import numpy as np
import cv2
import os
import pickle
from tqdm import tqdm

from datasets.ref_sampler.BaseRefSampler import BaseRefSampler
from datasets.BaseDataLoader import BaseDataLoader
from datasets.struct.data import Group, Sample
from r3d3.data_readers.rgbd_utils import compute_distance_matrix_flow, pose_matrix_to_quaternion


def get_depth(filename: str):
    """Get depth map from filename"""
    return cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / 100.


class FlowRefSampler(BaseRefSampler):
    def __init__(
            self,
            n_frames: int,
            cache_graph: Optional[bool] = True,
            load_cached_graph: Optional[bool] = True,
            scale: Optional[int] = 16,
            max_flow: Optional[float] = 256.,
            fmin: float = 8.0,
            fmax: float = 75.0,
            **kwargs,
    ):
        super(FlowRefSampler, self).__init__(**kwargs)
        self.n_frames = n_frames
        self.graph = None
        self.cache_graph = cache_graph
        self.load_cached_graph = load_cached_graph
        self.scale = scale
        self.max_flow = max_flow
        self.fmin = fmin
        self.fmax = fmax
        self.cache_graph_file = 'droid_graph_cache.pickle'
        self.rng = np.random.default_rng(2022)  # For consistent sampling across gpus and rep. results

    def initialize(
            self,
            dataset: Dict[str, List[Group]],
            dataset_path: str,
            cameras: Union[List[str], List[int]],
            dataloader: Type[BaseDataLoader],
            split: Optional[Union[List[str], Dict[str, List[int]]]] = None,
    ) -> Dict[str, List]:
        # ToDo: Check for multi-gpu usage: Multiple instances of dataloaders????
        cache_file = os.path.join(dataset_path, self.cache_graph_file)
        if self.load_cached_graph and os.path.isfile(cache_file):
            with open(cache_file, 'rb') as handle:
                self.graph = pickle.load(handle)
        else:
            self.graph = dict()
            for scene, groups in tqdm(dataset.items(), desc='Build scene graphs'):
                self.graph[scene] = {}
                for cam in tqdm(groups[0]['frames'].keys(), desc='Build cam graphs', leave=False):
                    poses = [pose_matrix_to_quaternion(group['frames'][cam]['pose']) for group in groups]
                    intrinsics = [group['frames'][cam]['intrinsics'] for group in groups]
                    depths = [os.path.join(dataset_path, group['frames'][cam]['depth']) for group in groups]
                    self.graph[scene][cam] = self._build_frame_graph(
                        poses=np.stack(poses),
                        intrinsics=np.stack(intrinsics),
                        depths=depths,
                        depth_reader=dataloader.get_depth
                    )
            if self.cache_graph:
                with open(cache_file, 'wb') as handle:
                    pickle.dump(self.graph, handle, protocol=pickle.HIGHEST_PROTOCOL)

        cam = cameras[0]
        self.sample_list: List[Sample] = []
        sample_idx_list: Dict[str, List] = {}
        for scene, groups in dataset.items():
            if (split is not None) and (scene not in split):
                continue
            sample_idx_list[scene] = []
            for idx in range(len(groups)):
                if (split is not None) and (type(split) is dict) and (idx not in split[scene]):
                    continue
                inds = [idx]
                while len(inds) < self.n_frames:
                    # get other frames within flow threshold
                    k = (self.graph[scene][cam][idx][1] > self.fmin) & (self.graph[scene][cam][idx][1] < self.fmax)
                    frames = self.graph[scene][cam][idx][0][k]

                    # prefer frames forward in time
                    if np.count_nonzero(frames[frames > idx]):
                        idx = self.rng.choice(frames[frames > idx])
                    elif np.count_nonzero(frames):
                        idx = self.rng.choice(frames)

                    inds += [idx]
                sample_idx_list[scene].append({i: int(j) for i, j in enumerate(inds)})
                sample: Sample = {
                    'sequence': scene,
                    'groups': {i: dataset[scene][j] for i, j in enumerate(inds)}
                }
                if len(set(inds)) == self.n_frames:
                    self.sample_list.append(sample)
        return sample_idx_list

    def _build_frame_graph(
        self,
        poses: np.ndarray,
        intrinsics: np.ndarray,
        depths: List[str],
        depth_reader: Callable[[str], np.ndarray]
    ) -> Dict[int, Tuple[int, float]]:
        """ Adapted from https://github.com/princeton-vl/DROID-SLAM/blob/main/droid_slam/data_readers/base.py#L69
        compute optical flow distance between all pairs of frames
        """
        def read_disp(filename):
            depth = depth_reader(filename)[self.scale // 2::self.scale, self.scale // 2::self.scale]
            depth[depth < 0.01] = np.mean(depth)
            return 1.0 / depth

        intrinsics = intrinsics / self.scale

        disps = np.stack(list(map(read_disp, depths)), 0)
        d = self.scale * compute_distance_matrix_flow(poses, disps, intrinsics)

        graph = {}
        for i in range(d.shape[0]):
            j, = np.where(d[i] < self.max_flow)
            graph[i] = (j, d[i, j])

        return graph
