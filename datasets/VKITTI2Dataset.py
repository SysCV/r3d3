from abc import ABC

from PIL.Image import Image
import numpy as np
import cv2
import csv
from tqdm import tqdm

from typing import Dict, List, Tuple, Optional
import os
from glob import glob

from .struct.data import Group, Frame, Boxes3D, MetaData
from .BaseDataset import BaseDataset
from vidar.utils.decorators import iterate1
from vidar.utils.read import read_image


class VKITTI2Dataset(BaseDataset, ABC):
    def __init__(self, **kwargs):
        super(VKITTI2Dataset, self).__init__(**kwargs)

    def _check_labels(self, labels: List[str]):
        provided_labels = ['rgb', 'depth', 'input_depth', 'pose', 'instance', 'optical_flow', 'scene_flow', 'boxes3d']
        if not all([label in provided_labels for label in labels]):
            raise ValueError("Only labels {} are provided by the VKITTI2 dataset.".format(provided_labels))

    @staticmethod
    def __get_basename(paths: List[str]) -> List[str]:
        return [os.path.basename(os.path.dirname(path)) for path in paths]

    def __get_dataset_structure(self) -> Tuple[Dict[str, List[str]], List[str], List[str], List[str]]:
        """ Creates Dict and List of available files
        Returns:
            rgb_tree: Dict. of all rgb frames from 'clone' mode and 'Camera_0', i.e. {'scene': List[path]}
            modes: List of all available modes ('clone', 'sunny', ...)
            modalities: List of all available sensor modalities (e.g. 'rgb', 'depth', ...)
            cameras: List of all available cameras (e.g. 'Camera_1')
        """
        scenes = glob(os.path.join(self.path, "*/"))
        modes = glob(os.path.join(self.path, scenes[0], "*/"))
        modalities = glob(os.path.join(modes[0], "frames/*/"))
        cameras = glob(os.path.join(modalities[0], "*/"))

        scenes = NewVKITTI2Dataset.__get_basename(scenes)
        modes = NewVKITTI2Dataset.__get_basename(modes)
        modalities = NewVKITTI2Dataset.__get_basename(modalities)
        cameras = NewVKITTI2Dataset.__get_basename(cameras)

        scenes.sort()
        modes.sort()
        modalities.sort()
        cameras.sort()

        rgb_tree = dict()
        for scene in scenes:
            rgb_tree[scene] = glob(os.path.join(self.path, scene, "clone/frames/rgb/Camera_0/*.jpg"))
            rgb_tree[scene].sort()
            rgb_tree[scene] = [os.path.relpath(rgb_path, self.path) for rgb_path in rgb_tree[scene]]    # make path rel.

        return rgb_tree, modes, modalities, cameras

    @staticmethod
    def __combine_boxes3d(boxes3d: Dict[str, Dict[str, Dict[int, Dict]]]) -> Dict[int, Dict]:
        """ Combines 3D boxes from different cameras and modes to single assembly of boxes for each time step
        Input:
            boxes3d: {'clone': {'Camera_0': {0: {0: {'dim': [...], ... }, ...}, ...}, ...}
        """
        combined_boxes3d = dict()
        for mode in boxes3d.keys():
            for cam in boxes3d[mode].keys():
                for frame, frame_boxes3d in boxes3d[mode][cam].items():
                    if frame not in combined_boxes3d:
                        combined_boxes3d[frame] = dict()
                    combined_boxes3d[frame].update(frame_boxes3d)
        return combined_boxes3d

    def _get_dataset(self) -> Dict[str, List[Group]]:
        """
        Returns: Dictionary of data samples in self.path, i.e. {'scene': [Group]}
        """
        rgb_tree, modes, modalities, cameras = self.__get_dataset_structure()

        dataset = dict()
        for scene, frames in tqdm(rgb_tree.items(), desc='process scenes'):
            groups = []
            intrinsics, poses, boxes3d = {}, {}, {}
            for mode in modes:
                intrinsics[mode] = self._get_intrinsics(os.path.join(self.path, scene, mode, 'intrinsic.txt'))
                poses[mode] = self._get_pose(os.path.join(self.path, scene, mode, 'extrinsic.txt'))
                boxes3d[mode] = self._get_boxes3d(os.path.join(self.path, scene, mode, 'pose.txt'))
            combined_boxes3d = self.__combine_boxes3d(boxes3d)
            for rgb_frame in tqdm(frames, desc='process frames', leave=False):
                group_id = int(rgb_frame[-9:-4])
                frames = dict()
                for mode in modes:
                    for cam in cameras:
                        rgb = None
                        if 'rgb' in modalities:
                            rgb = os.path.join(scene, mode, 'frames/rgb', cam,
                                               'rgb_{:05d}.jpg'.format(group_id))
                        depth = None
                        if 'depth' in modalities:
                            depth = os.path.join(scene, mode, 'frames/depth', cam,
                                                 'depth_{:05d}.png'.format(group_id))
                        instance_map = None
                        if 'instanceSegmentation' in modalities:
                            instance_map = os.path.join(scene, mode, 'frames/instanceSegmentation', cam,
                                                        'instancegt_{:05d}.png'.format(group_id))
                        fwd_opt_flow = None
                        if 'forwardFlow' in modalities:
                            fwd_opt_flow = self._if_exists(os.path.join(scene, mode, 'frames/forwardFlow', cam,
                                                                        'flow_{:05d}.png'.format(group_id)))
                        bwd_opt_flow = None
                        if 'backwardFlow' in modalities:
                            bwd_opt_flow = self._if_exists(os.path.join(scene, mode, 'frames/backwardFlow', cam,
                                                                        'backwardFlow_{:05d}.png'.format(group_id)))
                        fwd_scene_flow = None
                        if 'forwardSceneFlow' in modalities:
                            fwd_scene_flow = self._if_exists(os.path.join(scene, mode, 'frames/forwardSceneFlow', cam,
                                                                          'sceneFlow_{:05d}.png'.format(group_id)))
                        bwd_scene_flow = None
                        if 'backwardSceneFlow' in modalities:
                            bwd_scene_flow = self._if_exists(os.path.join(scene, mode, 'frames/backwardSceneFlow', cam,
                                                             'backwardSceneFlow_{:05d}.png'.format(group_id)))
                        metadata: MetaData = {
                            'url': rgb_frame,
                            'video_name': scene,
                            'index': group_id,
                        }

                        frame: Frame = {
                            'metadata': metadata,
                            'rgb': rgb,
                            'depth': depth,
                            'fwd_opt_flow': fwd_opt_flow,
                            'bwd_opt_flow': bwd_opt_flow,
                            'fwd_scene_flow': fwd_scene_flow,
                            'bwd_scene_flow': bwd_scene_flow,
                            'intrinsics': intrinsics[mode][cam][group_id],
                            'pose': poses[mode][cam][group_id],
                            'instance_map': instance_map,
                        }
                        frames[mode + '_' + cam] = frame

                metadata: MetaData = {
                    'video_name': scene,
                    'index': group_id,
                }
                if group_id in combined_boxes3d:
                    frame_boxes3d: Boxes3D = {
                        'idx': np.array([[idx] for idx in combined_boxes3d[group_id].keys()]),
                        'dim': np.array([box['dim'] for box in combined_boxes3d[group_id].values()]),
                        'pos': np.array([box['pos'] for box in combined_boxes3d[group_id].values()]),
                        'rot': np.array([box['rot'] for box in combined_boxes3d[group_id].values()]),
                    }
                else:
                    frame_boxes3d: Boxes3D = {
                        'idx': np.empty((0, 1), dtype=np.int32),
                        'dim': np.empty((0, 3), dtype=np.float32),
                        'pos': np.empty((0, 3), dtype=np.float32),
                        'rot': np.empty((0, 3), dtype=np.float32),
                    }
                group: Group = {
                    'metadata': metadata,
                    'boxes3d': frame_boxes3d,
                    'frames': frames,
                }
                groups += [group]
            dataset[scene] = groups
        return dataset

    def _split_dataset(self):
        """ There is no dataset split implemented at the moment (no official split available for VKITTI2 """
        pass

    @staticmethod
    @iterate1
    def _get_image(filename: str) -> Image:
        return read_image(filename)

    @staticmethod
    @iterate1
    def _get_depth(filename: str) -> np.ndarray:
        return cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / 100.

    @staticmethod
    @iterate1
    def _get_intrinsics(filename: str, cameras: Optional[List[str]] = None) -> Dict[str, Dict[int, np.ndarray]]:
        cameras = cameras if cameras is not None else ['Camera_0', 'Camera_1']
        # Open intrinsic file
        intrinsics = {cam: {} for cam in cameras}
        with open(filename, 'r') as f:
            # Get intrinsic parameters
            lines = list(csv.reader(f, delimiter=' '))[1:]
            for line in lines:
                frame = int(line[0])
                cam = cameras[int(line[1])]
                params = [float(param) for param in line[2:]]
                intrinsics[cam][frame] = np.array(params)
        # Return intrinsics
        return intrinsics

    @staticmethod
    @iterate1
    def _get_pose(filename: str, cameras: Optional[List[str]] = None) -> Dict[str, Dict[int, np.ndarray]]:
        cameras = cameras if cameras is not None else ['Camera_0', 'Camera_1']
        # Open extrinsics file
        poses = {cam: {} for cam in cameras}
        with open(filename, 'r') as f:
            # Get pose parameters
            lines = list(csv.reader(f, delimiter=' '))[1:]
            for line in lines:
                frame = int(line[0])
                cam = cameras[int(line[1])]
                poses[cam][frame] = np.linalg.inv(np.array([float(param) for param in line[2:]]).reshape(4, 4))
        # Return poses
        return poses

    # @staticmethod
    # def _get_ontology(filename, mode):
    #     """Get ontology from filename"""
    #     # Get ontology filename
    #     filename_idx = filename.rfind(mode) + len(mode)
    #     filename_ontology = os.path.join(filename[:filename_idx].replace(
    #         '/classSegmentation/', '/textgt/'), 'colors.txt')
    #     # Open ontology file
    #     with open(filename_ontology, 'r') as f:
    #         # Get ontology parameters
    #         lines = list(csv.reader(f, delimiter=' '))[1:]
    #         from collections import OrderedDict
    #         ontology = OrderedDict()
    #         for i, line in enumerate(lines):
    #             ontology[i] = {
    #                 'name': line[0],
    #                 'color': np.array([int(clr) for clr in line[1:]])
    #             }
    #     return ontology

    #  def _get_semantic(self, filename):
    #      """Get semantic from filename"""
    #      # Get semantic color map
    #      semantic_color = {key: np.array(val) for key, val in read_image(filename).items()}
    #      # Return semantic id map
    #      semantic_id = {key: semantic_color_to_id(val, self.ontology) for key, val in semantic_color.items()}
    #      return convert_ontology(semantic_id, self.ontology_convert)

    @staticmethod
    def _get_instance(filename: str) -> np.ndarray:
        return np.array(read_image(filename), dtype=np.int32) - 1    # background = -1, otw. = inst. id

    @staticmethod
    def _get_boxes3d(filename: str, cameras: Optional[List[str]] = None) -> Dict[str, Dict[int, Boxes3D]]:
        cameras = cameras if cameras is not None else ['Camera_0', 'Camera_1']
        boxes3d = {cam: {} for cam in cameras}
        prev_cam, prev_frame = -1, -1
        with open(filename, 'r') as file:
            for i, f in enumerate(file):
                if i == 0:
                    continue
                line = [float(a) for a in f.split(' ')]
                frame = int(line[0])
                cam = cameras[int(line[1])]
                idx = int(line[2])

                if prev_cam != cam or prev_frame != frame:
                    boxes3d[cam][frame] = {}
                boxes3d[cam][frame].update({idx: {
                    'dim': [line[6], line[5], line[4]],
                    'pos': line[7:10],
                    'rot': [line[11], line[10], line[12]],
                }})

                prev_cam = cam
                prev_frame = frame
        return boxes3d

    @staticmethod
    @iterate1
    def _get_instance_info(filename: str) -> Dict[int, Dict[str, str]]:
        instance_info = dict()
        with open(filename, 'r') as f:
            lines = list(csv.reader(f, delimiter=' '))[1:]
            for line in lines:
                track_id = int(line[0])
                instance_info[track_id] = {'label': line[1], 'model': line[2], 'color': line[3]}
        return instance_info

    @staticmethod
    @iterate1
    def _get_optical_flow(filename: str) -> Optional[np.ndarray]:
        # Return None if file does not exist
        if not os.path.exists(filename):
            return None
        else:
            # Get optical flow
            optical_flow = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            h, w = optical_flow.shape[:2]
            # Get invalid optical flow pixels
            invalid = optical_flow[..., 0] == 0
            # Normalize and scale optical flow values
            optical_flow = 2.0 / (2 ** 16 - 1.0) * optical_flow[..., 2:0:-1].astype('f4') - 1.
            optical_flow[..., 0] *= w - 1
            optical_flow[..., 1] *= h - 1
            # Remove invalid pixels
            optical_flow[invalid] = 0
            return optical_flow

    @staticmethod
    @iterate1
    def _get_scene_flow(filename: str) -> Optional[np.ndarray]:
        # Return None if file does not exist
        if not os.path.exists(filename):
            return None
        else:
            # Get scene flow
            scene_flow = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            # Return normalized and scaled optical flow (-10m to 10m)
            return (scene_flow[:, :, ::-1] * 2. / 65535. - 1.) * 10.


if __name__ == '__main__':
    boxes3d = NewVKITTI2Dataset._get_boxes3d('/media/datadrive/vkitti2/Scene01/15-deg-left/pose.txt')
    opt_flow = NewVKITTI2Dataset._get_optical_flow('/media/datadrive/vkitti2/Scene01/15-deg-left/frames/forwardFlow/Camera_0/flow_00000.png')
    scene_flow = NewVKITTI2Dataset._get_scene_flow('/media/datadrive/vkitti2/Scene01/15-deg-left/frames/forwardSceneFlow/Camera_0/sceneFlow_00000.png')
    depth = NewVKITTI2Dataset._get_depth('/media/datadrive/vkitti2/Scene01/15-deg-left/frames/depth/Camera_0/depth_00000.png')
    image = read_image('/media/datadrive/vkitti2/Scene01/15-deg-left/frames/rgb/Camera_0/rgb_00000.jpg')
    inst_seg = NewVKITTI2Dataset._get_instance('/media/datadrive/vkitti2/Scene01/15-deg-left/frames/instanceSegmentation/Camera_0/instancegt_00000.png')
    intrinsics = NewVKITTI2Dataset._get_intrinsics('/media/datadrive/vkitti2/Scene01/15-deg-left/intrinsic.txt')
    poses = NewVKITTI2Dataset._get_pose('/media/datadrive/vkitti2/Scene01/15-deg-left/extrinsic.txt')
    inst_info = NewVKITTI2Dataset._get_instance_info('/media/datadrive/vkitti2/Scene01/15-deg-left/info.txt')
    print("Test")
