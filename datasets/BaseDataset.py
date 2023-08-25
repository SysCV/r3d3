from abc import ABC
from typing import Optional, List, Dict, Union, Callable

import os
from torch.utils.data import Dataset
import numpy as np
import pickle

from vidar.utils.types import is_list
from vidar.datasets.utils.misc import stack_sample
from vidar.utils.config import Config, cfg_has, load_class

from datasets.utils.utils import intrinsics_4_to_nxn
from datasets.BaseDataLoader import BaseDataLoader
from datasets.struct.data import Group, InputData, InputFrame
from datasets.ref_sampler.DefaultRefSampler import DefaultRefSampler


class BaseDataset(Dataset, BaseDataLoader, ABC):
    """ BaseDataset compatible with vidar framework and allowing for more diverse features """
    dataset = {}

    def __init__(
            self,
            cfg: Config,
            path: str,
            cameras: List[str],
            labels:  Optional[List[str]] = None,
            labels_context: Optional[List[str]] = None,
            data_transform: Optional[Callable[[List[InputFrame]], List[InputFrame]]] = None,
            split: str = None,
            cache: Optional[bool] = True,
            load_cache: Optional[bool] = True,
            virtual: Optional[bool] = False,
            squeeze_sensor: Optional[bool] = True,
            n_objects: Optional[int] = 40,
            load_rel_pose: Optional[bool] = False,
            **kwargs
    ):
        """
        Args:
            cfg: Vidar config
            path: Path to dataset
            cameras: List of cameras to load
            labels: List of target labels to load - availability will be checked in _check_labels
            labels_context: List of context labels to load - availability will be checked in _check_labels
            data_transform: Data transformations,
            split: Dataset split - to be used in _split_dataset
            cache: True - dataset will be cached in a pickle file,
            load_cache: True - dataset will be loaded from pickle file, False - dataset will be built from scratch
            virtual: True - dataset is virtual, False - dataset is real-world
            squeeze_sensor: True - dimensions will be squeezed,
            n_objects: Number of objects which should be loaded,
            load_rel_pose: True - poses will be loaded relative to the first camera in 'cameras'
        """
        super().__init__()
        self.path = path
        self.labels = labels if labels is not None else []
        self.labels_context = labels_context if labels_context is not None else []

        self.cameras = cameras
        self.data_transform = data_transform
        self.split = split
        self.load_rel_pose = load_rel_pose

        self.num_cameras = len(cameras) if is_list(cameras) else cameras

        self.virtual = virtual
        self.squeeze_sensor = squeeze_sensor
        self.n_objects = n_objects

        self._check_labels(self.labels)
        self._check_labels(self.labels_context)

        # Create reference view sampler
        if cfg_has(cfg, 'ref_sampler'):
            self.ref_sampler = load_class(cfg.ref_sampler.name, 'datasets/ref_sampler')(**cfg.ref_sampler.dict)
        else:
            self.ref_sampler = DefaultRefSampler()

        # Load dataset: 1. Use already loaded dataset, 2. Use cached dataset, 3. Build form scratch
        if self.__class__.__name__ in BaseDataset.dataset:
            self.dataset = BaseDataset.dataset[self.__class__.__name__]
        else:
            dataset_cache = os.path.join(path, f'cached_{self.__class__.__name__}.pickle')
            if os.path.isfile(dataset_cache) and load_cache:
                with open(dataset_cache, 'rb') as pickle_file:
                    self.dataset = pickle.load(pickle_file)
            else:
                print("=== Pre-Process Dataset ===")
                self.dataset = self._get_dataset()
                if cache:
                    with open(dataset_cache, 'wb') as pickle_file:
                        pickle.dump(self.dataset, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
            BaseDataset.dataset[self.__class__.__name__] = self.dataset

        dataset_split = None
        if self.split is not None:
            dataset_split = self._split_dataset()

        self.ref_sampler.initialize(
            dataset=self.dataset,
            dataset_path=self.path,
            cameras=self.cameras,
            dataloader=self.__class__,
            split=dataset_split
        )

    def _get_dataset(self) -> Dict[str, List[Group]]:
        """ Creates dataset from scratch, result might be buffered """
        raise NotImplementedError

    def _split_dataset(self) -> Union[List[str], Dict[str, List[str]]]:
        """
        Returns:
            dataset_split: List of scenes in self.split / Dict of {scene: [frames]} frame names
        """
        raise NotImplementedError

    def _check_labels(self, labels: List[str]):
        """ Check wether given labels are provided by dataset! Raise exception if not """
        raise NotImplementedError

    def _if_exists(self, file: str) -> Optional[str]:
        """ Returns given path to file if file exists """
        return file if os.path.isfile(os.path.join(self.path, file)) else None

    def __len__(self) -> int:
        return len(self.ref_sampler)

    def __getitem__(self, idx: int) -> InputData:
        sample_data = self.ref_sampler[idx]
        input_frames: List[InputFrame] = []

        # Add frame elements
        for camera in self.cameras:
            input_frame: InputFrame = {
                'idx': idx,
                'frame_idx': {},
                'cam': camera,
                'filename': {},
                'rgb': {},
                'intrinsics': {},
            }
            for ctx, group in sample_data['groups'].items():
                frame = group['frames'][camera]
                rgb_path = os.path.join(self.path, frame['rgb'])
                input_frame['frame_idx'].update({ctx: group['metadata']['frame_index']})
                input_frame['filename'].update({ctx: frame['rgb'].replace('.png', '').replace('.jpg', '')})
                input_frame['rgb'].update({ctx: self.get_image(rgb_path)})
                input_frame['intrinsics'].update({ctx: intrinsics_4_to_nxn(frame['intrinsics'], dim=3)})

                if self.with_pose(ctx != 0):
                    if 'pose' not in input_frame:
                        input_frame['pose'] = {}
                    input_frame['pose'][ctx] = frame['pose']
                if self.with_depth(ctx != 0):
                    if 'depth' not in input_frame:
                        input_frame['depth'] = {}
                    input_frame['depth'][ctx] = self.get_depth(os.path.join(self.path, frame['depth']))
                if self.with_mask(ctx != 0):
                    if 'mask' not in input_frame:
                        input_frame['mask'] = {}
                    if isinstance(frame['mask'], str):
                        input_frame['mask'][ctx] = self.get_mask(os.path.join(self.path, frame['mask']))
                    else:
                        input_frame['mask'][ctx] = frame['mask']
                if self.with_instance(ctx != 0):
                    if 'instance' not in input_frame:
                        input_frame['instance'] = {}
                    input_frame['instance'][ctx] = self.get_instance(os.path.join(self.path, frame['instance_map']))
                if self.with_optical_flow(ctx != 0):
                    if 'bwd_optical_flow' not in input_frame:
                        input_frame['bwd_optical_flow'] = {}
                        input_frame['fwd_optical_flow'] = {}
                    if frame['bwd_opt_flow'] is not None:
                        input_frame['bwd_optical_flow'][ctx] = self.get_optical_flow(
                            os.path.join(self.path, frame['bwd_opt_flow'])
                        )
                    if frame['fwd_opt_flow'] is not None:
                        input_frame['fwd_optical_flow'][ctx] = self.get_optical_flow(
                            os.path.join(self.path, frame['fwd_opt_flow'])
                        )
                if self.with_optical_flow_mask(ctx != 0):
                    if 'flow_mask' not in input_frame:
                        input_frame['bwd_flow_mask'] = {}
                        input_frame['fwd_flow_mask'] = {}
                    if frame['bwd_opt_flow_mask'] is not None:
                        input_frame['bwd_flow_mask'][ctx] = self.get_optical_flow_mask(os.path.join(
                            self.path, frame['bwd_opt_flow_mask']))
                    if frame['fwd_opt_flow_mask'] is not None:
                        input_frame['fwd_flow_mask'][ctx] = self.get_optical_flow_mask(os.path.join(
                            self.path, frame['fwd_opt_flow_mask']))
                if self.with_scene_flow(ctx != 0):
                    if 'bwd_scene_flow' not in input_frame:
                        input_frame['bwd_scene_flow'] = {}
                        input_frame['fwd_scene_flow'] = {}
                    if frame['bwd_scene_flow'] is not None:
                        input_frame['bwd_scene_flow'][ctx] = self.get_scene_flow(
                            os.path.join(self.path, frame['bwd_scene_flow'])
                        )
                    if frame['fwd_scene_flow'] is not None:
                        input_frame['fwd_scene_flow'][ctx] = self.get_scene_flow(
                            os.path.join(self.path, frame['fwd_scene_flow'])
                        )
                if self.with_semantic(ctx != 0):
                    if 'semantic' not in input_frame:
                        input_frame['semantic'] = {}
                    input_frame['semantic'][ctx] = self.get_semantic(os.path.join(self.path, frame['semantic_map']))

                if self.with_others(ctx != 0):
                    if 'others' not in input_frame:
                        input_frame['others'] = {}
                    labels = self.labels_context if ctx != 0 else self.labels
                    input_frame['others'][ctx] = self.get_others(self.path, frame['others'], labels)
            input_frames.append(input_frame)

        # Calc poses frame2cam0t0
        if self.load_rel_pose:
            ref_pose = input_frames[0]['pose'][0]
            for i in range(len(input_frames)):
                for ctx in input_frames[i]['pose']:
                    input_frames[i]['pose'][ctx] = np.linalg.inv(ref_pose) @ input_frames[i]['pose'][ctx]

        # Do Frame Transformations
        if self.data_transform:
            input_frames = self.data_transform(input_frames)

        input_data: InputData = stack_sample(sample=input_frames, squeeze_sensor=self.squeeze_sensor)
        input_data['scene'] = sample_data['sequence']  # Add Scene info

        # Add group elements
        for ctx, group in sample_data['groups'].items():
            if self.with_boxes3d(ctx == 0):
                if 'bboxes3d' not in input_data:
                    input_data['bboxes3d'] = {}
                if len(group['boxes3d']['idx']) < self.n_objects:
                    input_data['bboxes3d'][ctx] = {}
                    for key, val in group['boxes3d'].items():
                        input_data['bboxes3d'][ctx][key] = np.zeros((self.n_objects, *val.shape[1:]))
                        if key == 'idx':
                            input_data['bboxes3d'][ctx]['idx'] -= 1.
                        input_data['bboxes3d'][ctx][key][:len(group['boxes3d'][key])] = group['boxes3d'][key]
                else:
                    input_data['bboxes3d'][ctx] = {key: val[:self.n_objects] for key, val in group['boxes3d'].items()}
            if self.with_points(ctx == 0):
                if 'points' not in input_data:
                    input_data['points'] = {}
                input_data['points'] = self.get_points(os.path.join(self.path, group['lidar']))
        return input_data

    def with_depth(self, context: Optional[bool] = False) -> bool:
        """If dataset contains depth"""
        return 'depth' in self.labels_context if context else 'depth' in self.labels

    def with_points(self, context: Optional[bool] = False) -> bool:
        """If dataset contains point-cloud data"""
        return 'points' in self.labels_context if context else 'points' in self.labels

    def with_mask(self, context: Optional[bool] = False) -> bool:
        """If dataset contains mask"""
        return 'mask' in self.labels_context if context else 'mask' in self.labels

    def with_pose(self, context: Optional[bool] = False) -> bool:
        """If dataset contains pose"""
        return 'pose' in self.labels_context if context else 'pose' in self.labels

    def with_semantic(self, context: Optional[bool] = False) -> bool:
        """If dataset contains semantic"""
        return 'semantic' in self.labels_context if context else 'semantic' in self.labels

    def with_instance(self, context: Optional[bool] = False) -> bool:
        """If dataset contains instance"""
        return 'instance' in self.labels_context if context else 'instance' in self.labels

    def with_optical_flow(self, context: Optional[bool] = False) -> bool:
        """If dataset contains optical flow"""
        return 'optical_flow' in self.labels_context if context else 'optical_flow' in self.labels

    def with_scene_flow(self, context: Optional[bool] = False) -> bool:
        """If dataset contains scene flow"""
        return 'scene_flow' in self.labels_context if context else 'scene_flow' in self.labels

    def with_boxes2d(self, context: Optional[bool] = False) -> bool:
        """If dataset contains 2d bounding boxes"""
        return 'boxes2d' in self.labels_context if context else 'boxes2d' in self.labels

    def with_boxes3d(self, context: Optional[bool] = False) -> bool:
        """If dataset contains 3d bounding boxes"""
        return 'boxes3d' in self.labels_context if context else 'boxes3d' in self.labels

    def with_lidar(self, context: Optional[bool] = False) -> bool:
        """If dataset contains lidar"""
        return 'lidar' in self.labels_context if context else 'lidar' in self.labels

    def with_radar(self, context: Optional[bool] = False) -> bool:
        """If dataset contains radar"""
        return 'radar' in self.labels_context if context else 'radar' in self.labels

    def with_optical_flow_mask(self, context: Optional[bool] = False) -> bool:
        """ If dataset contains optical_flow_mask """
        return 'optical_flow_mask' in self.labels_context if context else 'optical_flow_mask' in self.labels

    def with_others(self, context: Optional[bool] = False) -> bool:
        """ If dataset contains others """
        if context:
            return any('others' in label for label in self.labels_context)
        else:
            return any('others' in label for label in self.labels)
