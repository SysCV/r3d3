from abc import ABC
from typing import Dict, List, Tuple
import os

from PIL.Image import Image
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from scipy.spatial.transform import Rotation

from scalabel.label.from_nuscenes import load_data, parse_sequence
from scalabel.label.utils import get_matrix_from_extrinsics
from scalabel.label.typing import Label, Box3D
from nuscenes.utils.splits import create_splits_scenes
from vidar.utils.decorators import iterate1
from vidar.utils.read import read_image, read_txt, read_json

from datasets.struct.data import Group, Frame, MetaData, Boxes3D
from datasets.BaseDataset import BaseDataset
from datasets.utils.utils import check_depth_maps


def box_to_matrix(box: Box3D):  # ToDo: Check!!!
    matrix = np.eye(4)
    matrix[0:3, 0:3] = Rotation.from_euler('XYZ', box.orientation).as_matrix()
    matrix[0:3, 3] = box.location
    return matrix


def matrix_to_pos_rot(matrix) -> Tuple[np.ndarray, np.ndarray]:  # ToDo: Check!!!
    pos = matrix[0:3, 3]
    rot = Rotation.from_matrix(matrix[0:3, 0:3]).as_euler('xyz')
    return pos, rot


class NuScenesDataset(BaseDataset, ABC):
    """ NuScenes Dataset https://www.nuscenes.org/
    Requires the following dataset structure:
    nuScenes    // Download from here: https://www.nuscenes.org/nuscenes#download
    ├ $SCENE_DIR
        ├ samples
            ├ CAM_$CAM_DIR
                └ xyz.jpg
            └ LIDAR_TOP
                └ xyz.pcd.bin
        ├ sweeps
            └ CAM_$CAM_DIR
                └ xyz.jpg
        ├ mask
            └ $CAM.png      // self-occlusion mask - 255: Not occluded - 0: occluded
        └ v1.0-trainval
            └ ...
    """
    def __init__(self, version: str, **kwargs):
        self.version = version
        self.id_map = {}
        self.keyframes_only = True  # ToDo: Extend to all frames ??
        super(NuScenesDataset, self).__init__(**kwargs)

    def _check_labels(self, labels: List[str]):
        provided_labels = ['rgb', 'depth', 'pose', 'mask', 'lidar', 'boxes3d']
        if not all([label in provided_labels for label in labels]):
            raise ValueError("Only labels {} are provided by the NuScenes dataset.".format(provided_labels))

    def scalabel_id_to_id(self, nuscenes_id: str) -> int:
        """
        Assigns novel int id to scalabel str id
        """
        if nuscenes_id in self.id_map:
            return self.id_map[nuscenes_id]
        elif len(self.id_map) == 0:
            self.id_map[nuscenes_id] = 0
            return 0
        else:
            new_id = max(self.id_map.values()) + 1
            self.id_map[nuscenes_id] = new_id
            return new_id

    def _labels_to_3dbb(self, labels: List[Label], ref_pose: np.ndarray) -> Boxes3D:
        """
        Converts scalabel labels to 3d BB
        Input:
            labels: List of scalabel labels
            ref_pose: to_world transformation matrix -- (4, 4)
        """
        if labels is not None:
            n_boxes = len(labels)
            frame_boxes3d: Boxes3D = {
                'idx': np.zeros((n_boxes, 1), dtype=np.int32),
                'dim': np.zeros((n_boxes, 3), dtype=np.float32),
                'pos': np.zeros((n_boxes, 3), dtype=np.float32),
                'rot': np.zeros((n_boxes, 3), dtype=np.float32),
            }
            for i, label in enumerate(labels):
                box_matrix = box_to_matrix(label.box3d)
                pos_world, rot_world = matrix_to_pos_rot(ref_pose @ box_matrix)

                frame_boxes3d['idx'][i, 0] = self.scalabel_id_to_id(label.id)
                frame_boxes3d['dim'][i] = label.box3d.dimension   # ToDo Test if actually (w, h, l) and not (h, w, l)
                frame_boxes3d['pos'][i] = rot_world
                frame_boxes3d['rot'][i] = pos_world
        else:
            frame_boxes3d: Boxes3D = {
                'idx': np.zeros((0, 1), dtype=np.int32),
                'dim': np.zeros((0, 3), dtype=np.float32),
                'pos': np.zeros((0, 3), dtype=np.float32),
                'rot': np.zeros((0, 3), dtype=np.float32),
            }
        return frame_boxes3d

    def _get_dataset(self) -> Dict[str, List[Group]]:
        """
        Returns: Dictionary of data samples in self.path, i.e. {'scene': [Group]}
        """
        dataset = {}
        data, df = load_data(
            filepath=self.path,
            version=self.version,
        )

        masks = {}
        for cam_file in os.listdir(os.path.join(self.path, 'mask')):
            cam = cam_file.split('.')[0]
            masks[cam] = NuScenesDataset.get_mask(os.path.join(self.path, 'mask', cam_file))
        # Gather dataset
        for first_sample_token, scene_name in tqdm(zip(df.first_sample_token.values, df.scene_name.values)):
            frames, groups = parse_sequence(data, self.keyframes_only, (first_sample_token, scene_name))

            scene_data = {}
            for f in frames:
                cam = f.url.split('/')[1]

                intrinsics = np.array([
                    f.intrinsics.focal[0], f.intrinsics.focal[1],
                    f.intrinsics.center[0], f.intrinsics.center[1]
                ])
                # Due to bug in training of compl-network.
                intrinsics[[1, 3]] *= 0.964285  # Remove line, delete dataset cache and retrain for better results!

                pose = get_matrix_from_extrinsics(f.extrinsics)

                metadata: MetaData = {
                    'url': f.url,
                    'video_name': f.videoName if f.videoName is not None else '?',
                    'frame_index': f.frameIndex if f.frameIndex is not None else 0,
                    'timestamp': f.timestamp
                }
                rgb = f.url if f.url is not None else ''
                frame: Frame = {
                    'metadata': metadata,
                    'rgb': rgb,
                    'depth': rgb.replace('CAM', 'DEPTH').replace('.jpg', '.npz'),
                    'mask': masks[cam],
                    'intrinsics': intrinsics,
                    'pose': pose,
                    'instance_map': None,
                    'others': None
                }
                if f.frameIndex not in scene_data:
                    scene_data[f.frameIndex] = {}
                scene_data[f.frameIndex][cam] = frame

            for g in groups:
                if len(g.frames) < 6:
                    continue
                lidar_pose = get_matrix_from_extrinsics(g.extrinsics) if g.extrinsics is not None else None

                metadata: MetaData = {
                    'url': g.url,
                    'video_name': g.videoName if g.videoName is not None else '?',
                    'frame_index': g.frameIndex if g.frameIndex is not None else 0,
                    'timestamp': g.timestamp,
                }
                group: Group = {
                    'metadata': metadata,
                    'group_pose': None,
                    'lidar': g.url,
                    'lidar_pose': lidar_pose,
                    'boxes3d': self._labels_to_3dbb(g.labels, lidar_pose) if g.labels is not None else None,
                    'frames': scene_data[g.frameIndex],
                }
                if g.videoName not in dataset:
                    dataset[g.videoName] = []
                dataset[g.videoName].append(group)
        # create depth maps if they do not exist
        print('Check & create depth maps ...')
        check_depth_maps(path=self.path, dataset=dataset, data_loader=self.__class__, shape=(900, 1600))
        return dataset

    def _split_dataset(self):
        if self.split.endswith('.txt'):
            return read_txt(self.split)
        elif self.split.endswith('.json'):
            return read_json(self.split)
        else:
            return create_splits_scenes()[self.split]

    @staticmethod
    @iterate1
    def get_image(filename: str) -> Image:
        if not os.path.isfile(filename) and filename.endswith('.jpg'):
            filename = filename.replace('.jpg', '.png')
        if not os.path.isfile(filename) and filename.endswith('.png'):
            filename = filename.replace('.png', '.jpg')
        return read_image(filename)

    @staticmethod
    @iterate1
    def get_depth(filename: str) -> npt.NDArray[np.float32]:
        if os.path.isfile(filename):
            npz_data = np.load(filename)
            return npz_data[npz_data.files[0]].reshape(900, 1600)
        else:
            return np.zeros((900, 1600), dtype=np.float32)

    @staticmethod
    @iterate1
    def get_mask(filename: str) -> npt.NDArray[np.int32]:
        return np.array(read_image(filename), dtype=np.int32)[..., 0]

    @staticmethod
    @iterate1
    def get_points(filename: str) -> npt.NDArray[np.float32]:
        points = np.fromfile(filename, dtype=np.float32)
        return np.reshape(points, (-1, 5))[:, :3]
