from abc import ABC

from PIL.Image import Image
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
import cv2

from typing import Dict, List
import os

from datasets.struct.data import Group, Frame, MetaData
from datasets.BaseDataset import BaseDataset
from datasets.utils.utils import check_depth_maps
from vidar.utils.decorators import iterate1
from vidar.utils.read import read_txt, read_json, read_image

from dgp.datasets.synchronized_dataset import SynchronizedSceneDataset


class DDADDataset(BaseDataset, ABC):
    """ Dense Depth for Autonomous Driving (DDAD) dataset class - https://github.com/TRI-ML/DDAD
    Requires the following dataset structure:
    DDAD    // Download from here: https://tri-ml-public.s3.amazonaws.com/github/DDAD/datasets/DDAD.tar
    ├ $SCENE_DIR
        ├ rgb
            └ $CAM_DIR
                └ xxxxxxxxxxxxxxxxx.png
        ├ calibration
            └ xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.json
        ├ point_cloud
            └ $CAM_DIR
                └ xxxxxxxxxxxxxxxxx.npz
        ├ occl_mask // Download from here: https://cloud.tsinghua.edu.cn/f/c654cd272a6a42c885f9/?dl=1
            └ $CAM_DIR
                └ mask.png      // self-occlusion mask - 0: Not occluded - 255: occluded
        └ scene_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.json
            └ $CAM_DIR
                └ xxxxxxxxxxxxxxxxx.npz
    └ ddad.json
    """
    def __init__(self, **kwargs):
        super(DDADDataset, self).__init__(**kwargs)

    def _check_labels(self, labels: List[str]):
        provided_labels = ['rgb', 'depth', 'pose', 'instance', 'mask']
        if not all([label in provided_labels for label in labels]):
            raise ValueError("Only labels {} are provided by the DDAD dataset.".format(provided_labels))

    def _get_dataset(self) -> Dict[str, List[Group]]:
        """
        Returns: Dictionary of data samples in self.path, i.e. {'scene': [Group]}
        """
        dataset = dict()
        scene_split = dict()
        masks = {}  # Load one self-occlusion mask per camera & frame
        for split in tqdm(['train', 'val'], desc='Split'):
            scene_split[split] = []
            dataset_loader = SynchronizedSceneDataset(
                scene_dataset_json=os.path.join(self.path, 'ddad.json'),
                split=split,
                datum_names=('lidar', 'CAMERA_01', 'CAMERA_05', 'CAMERA_06', 'CAMERA_07', 'CAMERA_08', 'CAMERA_09'),
                requested_annotations=('bounding_box_2d', 'bounding_box_3d'),
                generate_depth_from_datum='lidar'
            )
            for scene_idx, scene in enumerate(tqdm(dataset_loader.scenes, desc='Scene', leave=False)):
                scene_name = scene.scene.name
                if scene_name not in masks:
                    masks[scene_name] = {}
                groups = []
                origin = None
                for idx, sample in enumerate(scene.samples):
                    lidar_datum = dataset_loader.get_datum(scene_idx, idx, 'lidar')
                    lidar: str = os.path.join(scene_name, lidar_datum.datum.point_cloud.filename)
                    lidar_pose = dataset_loader.get_datum_pose(lidar_datum).matrix
                    if origin is None:  # use lidar pose of first group as origin of coordinate system!
                        origin = lidar_pose.copy()
                    lidar_pose = np.linalg.inv(origin) @ lidar_pose
                    frames = dict()
                    for cam in ('camera_01', 'camera_05', 'camera_06', 'camera_07', 'camera_08', 'camera_09'):
                        datum = dataset_loader.get_datum(scene_idx, idx, cam)

                        calibration = dataset_loader.get_camera_calibration(
                            sample.calibration_key,
                            datum.id.name
                        )
                        intrinsics = np.array([calibration.fx, calibration.fy, calibration.cx, calibration.cy])
                        pose = np.linalg.inv(origin) @ dataset_loader.get_datum_pose(datum).matrix
                        rgb: str = os.path.join(scene_name, datum.datum.image.filename)
                        depth: str = rgb.replace('rgb', 'depth').replace('.png', '.npz')
                        if cam not in masks[scene_name]:
                            mask_path = os.path.join(os.path.dirname(rgb).replace('rgb', 'occl_mask'), 'mask.png')
                            if os.path.isfile(os.path.join(self.path, mask_path)):
                                mask = self.get_mask(os.path.join(self.path, mask_path)).astype(np.uint8)
                                masks[scene_name][cam] = cv2.resize(mask, (640, 384), interpolation=cv2.INTER_NEAREST)
                            else:
                                masks[scene_name][cam] = None
                        timestamp = datum.id.timestamp.ToMicroseconds()

                        metadata: MetaData = {
                            'url': os.path.join(self.path, rgb),
                            'video_name': scene_name,
                            'frame_index': idx,
                            'timestamp': timestamp,
                        }
                        frame: Frame = {
                            'metadata': metadata,
                            'rgb': rgb,
                            'depth': depth,
                            'mask': masks[scene_name][cam],
                            'intrinsics': intrinsics,
                            'pose': pose,
                            'instance_map': None,
                            'others': None
                        }
                        frames[cam] = frame
                    metadata: MetaData = {
                        'url': os.path.join(self.path, lidar),
                        'video_name': scene.scene.name,
                        'frame_index': idx,
                        'timestamp': None,
                    }
                    group: Group = {
                        'metadata': metadata,
                        'group_pose': None,
                        'lidar': lidar,
                        'lidar_pose': lidar_pose,
                        'boxes3d': None,
                        'frames': frames,
                    }
                    groups.append(group)
                scene_split[split].append(scene.scene.name)
                dataset[scene.scene.name] = groups
        # create depth maps if they do not exist
        print('Check & create depth maps ...')
        check_depth_maps(path=self.path, dataset=dataset, data_loader=self.__class__, shape=(1216, 1936))
        return dataset

    def _split_dataset(self):
        if self.split.endswith('.txt'):
            return read_txt(self.split)
        elif self.split.endswith('.json'):
            return read_json(self.split)
        else:
            if self.split == 'train':
                return ["{:06d}".format(i) for i in range(0, 150)]
            elif self.split == 'val':
                return ["{:06d}".format(i) for i in range(150, 200)]
            elif self.split == 'test':
                raise NotImplementedError('Test set not implemented for DDAD dataset!')
            return None

    @staticmethod
    @iterate1
    def get_image(filename: str) -> Image:
        return read_image(filename)

    @staticmethod
    @iterate1
    def get_depth(filename: str) -> npt.NDArray[np.float32]:
        npz_data = np.load(filename)
        return npz_data[npz_data.files[0]].reshape(1216, 1936)

    @staticmethod
    @iterate1
    def get_mask(filename: str) -> npt.NDArray[np.int32]:
        return 255 - np.array(read_image(filename), dtype=np.int32)[..., 0]

    @staticmethod
    @iterate1
    def get_points(filename: str) -> npt.NDArray[np.float32]:
        npz_data = np.load(filename)
        return npz_data['data'][:, :3].astype(np.float32)
