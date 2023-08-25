from __future__ import annotations
import sys
sys.path.append('thirdparty/vidar')

import torch
from typing import Tuple, List, Optional
from typing_extensions import TypedDict, NotRequired
from torch.utils.data import DataLoader
from vidar.utils.setup import setup_datasets


class EvalSample(TypedDict):
    """
    t: Sample timestamp
    images: Sample images of shape [C, 3, H, W] in [0, 1] where C is the # cameras
    intrinsics: Sample intrinsics of shape [C, 4] with [[fx, fy, cx, cy], ...]
    rel_poses: Relative poses camx2cam0 of shape [C, 7] with [[x, y, z, quat], ...]
    mask: Self-occlusion masks of shape [C, H, W] and type torch.bool where True = valid pixel
    poses: Absolute gt poses camx2world of shape [C, 7] with [[x, y, z, quat], ...]
    depth: Gt depth of shape [C, H, W] with unavailable depths = 0.0
    """
    timestamp: float
    images: torch.Tensor
    intrinsics: torch.Tensor
    rel_poses: torch.Tensor
    filename: List[float]
    mask: NotRequired[torch.Tensor | None]
    poses: NotRequired[torch.Tensor | None]
    depth: NotRequired[torch.Tensor | None]


class SampleIterator:
    """ Iterates through samples in a dataset scene"""
    def __init__(self, scene_iter, n_samples):
        self.scene_iter = scene_iter
        self.n_samples = n_samples
        self.ref_pose = None
        self.i = -1

    def __len__(self):
        return self.n_samples

    def __iter__(self):
        return self

    def __next__(self):
        self.i += 1
        if self.i == 0:
            return self.process_sample(self.scene_iter.first_sample)
        else:
            sample = next(self.scene_iter.data_iterator, None)
            scene = sample['scene'][0] if sample is not None else None
            if scene == self.scene_iter.scene:
                return self.process_sample(sample)
            else:
                self.scene_iter.scene = scene
                self.scene_iter.first_sample = sample
                self.ref_pose = None
                raise StopIteration

    def process_sample(self, sample):
        self.ref_pose = sample['pose'][0][0][0:1].clone() if self.ref_pose is None else self.ref_pose
        sample['pose'][0][0] = self.ref_pose.inverse() @ sample['pose'][0][0]
        return sample


class SceneIterator:
    """ Iterates through dataset scenes """
    def __init__(self, dataloader):
        self.data_iterator = iter(dataloader)
        self.scene_size = {scene: len(groups) for scene, groups in dataloader.dataset.dataset.items()}
        self.scene = -1
        self.first_sample = None

    # def __len__(self):
    #     return len(self.scene_size)

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[str, SampleIterator]:
        if self.scene == -1:
            self.first_sample = next(self.data_iterator, None)
            self.scene = self.first_sample['scene'][0] if self.first_sample is not None else None
        if self.first_sample is None:
            raise StopIteration
        return self.scene, SampleIterator(self, self.scene_size[self.scene])


def setup_dataloaders(cfg_path: str, n_workers: Optional[int] = 1) -> List[DataLoader]:
    """ Returns a list of torch dataloaders
    Args:
        cfg_path: Dataset config path
        n_workers: Number of torch dataloader workers
    Returns:
        dataloaders: List of torch datalaoders as defined in config
    """
    datasets = setup_datasets(cfg_path)[0]['validation']
    dataloaders = []
    for dataset in datasets:
        dataloaders.append(
            DataLoader(
                dataset,
                batch_size=1,
                pin_memory=True,
                num_workers=n_workers,
                shuffle=False,
            )
        )
    return dataloaders


if __name__ == '__main__':
    dls = setup_dataloaders(
        cfg_path='configs/evaluation/r3d3/r3d3_evaluation_ddad.yaml',
        n_workers=0
    )
    for dl in dls:
        for scene_name, scene_samples in SceneIterator(dl):
            print(scene_name)
            for sample in scene_samples:
                print(sample['t'])
