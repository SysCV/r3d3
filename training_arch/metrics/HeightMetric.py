from typing import Dict, List, Tuple, Optional

import os
import torch
import numpy as np

from vidar.metrics.base import BaseEvaluation
from vidar.utils.config import Config, cfg_has
from vidar.utils.data import dict_remove_nones
from vidar.utils.distributed import on_rank_0
from vidar.utils.logging import pcolor
from vidar.utils.types import is_dict, is_list
from vidar.utils.write import write_json
import pyransac3d as pyrsc

from training_arch.utils.utils import BackprojectDepth, intr_3x3_to_4x4


class HeightMetric(BaseEvaluation):
    """
    Detph evaluation metrics

    Parameters
    ----------
    cfg : Config
        Configuration file
    """
    def __init__(self, cfg: Config):
        """
        Args:
            cfg: Config which contains:
                height_threshold: Pixel height below ground-plane in meters which is considered as bad - default: 0.5
                ratio_threshold: Ratio of pixels which have to be bad s.t. the frame is considered as bad. Either
                    float (same threshold for cameras) or list of floats of length n_cams (each cam with own thresh)
                    - default: 0.1
                good_frames_split_path: Path to folder where good-file split should be saved. If none, files
                    won't be saved - default: None
        """
        super().__init__(
            cfg,
            name='height',
            task='height',
            metrics=('ratio', 'valid'),
        )
        self.height_threshold = cfg_has(cfg, 'height_threshold', .5)
        self.ratio_threshold = cfg_has(cfg, 'ratio_threshold', 0.1)
        self.good_frames_split_path = cfg_has(cfg, 'good_frames_split_path', None)
        assert isinstance(self.ratio_threshold, float) or isinstance(self.ratio_threshold, List), \
            f"'ratio_threshold' is expected to be of type float or List but is {type(self.ratio_threshold)}"

        self.backproject_depth = None
        self.good_frames: Dict[str, Dict[str: List[int]]] = {}   # {cam: {scene: [0, 1, ... (indices)]}}

    @staticmethod
    def reduce_fn(metrics, seen):
        """ Reduce function """
        valid = seen.view(-1) > 0
        return (metrics[valid] / seen.view(-1, 1)[valid]).mean(0)

    def populate_metrics_dict(self, metrics, metrics_dict, prefix):
        """ Populate metrics function """
        for metric in metrics:
            if metric.startswith(self.name):
                name, suffix = metric.split('|')
                for i, key in enumerate(self.metrics):
                    metrics_dict[f'{prefix}-{name}|{key}_{suffix}'] = \
                        metrics[metric][i].item()

    @on_rank_0
    def print(self, reduced_data, prefixes):
        """ Print function & save good-file split"""
        print()
        print(self.horz_line)
        print(self.metr_line.format(*((self.name.upper(),) + self.metrics)))
        for n, metrics in enumerate(reduced_data):
            if sum([self.name in key for key in metrics.keys()]) == 0:
                continue
            print(self.horz_line)
            print(self.wrap(pcolor('*** {:<114}'.format(prefixes[n]), **self.font1)))
            print(self.horz_line)
            for key, metric in sorted(metrics.items()):
                if self.name in key:
                    print(self.wrap(pcolor(self.outp_line.format(
                        *((key.upper(),) + tuple(metric.tolist()))), **self.font2)))
        print(self.horz_line)
        print()

        if self.good_frames_split_path is not None:
            self.save_good_file_split()

    def register_frame(self, cam, scene, idx):
        if cam not in self.good_frames:
            self.good_frames.update({cam: {}})
        if scene not in self.good_frames[cam]:
            self.good_frames[cam].update({scene: []})
        self.good_frames[cam][scene].append(idx)

    def save_good_file_split(self):
        for cam, good_frames in self.good_frames.items():
            path = os.path.join(self.good_frames_split_path, f'good_frames_split_{cam}.json')
            write_json(path, good_frames)

    def get_points(self, depth, intrinsics, pose=None):
        intr = intr_3x3_to_4x4(intrinsics).inverse()
        if self.backproject_depth is None:
            b, _, h, w = depth.shape
            self.backproject_depth = BackprojectDepth(b, h, w).to(depth.device)
        cam_points = self.backproject_depth(depth, intr)
        if pose is not None:
            points = pose @ cam_points
        else:
            points = cam_points
        return points.permute(0, 2, 1)[..., :3]

    def eval_points(
            self,
            points,
            normals,
            plane_heights,
            scenes: List[str],
            indices: torch.Tensor,
            cams: Tuple,
            cam_i: int,
            suffix: Optional[str] = '',
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            points: 3D points of shape [B, N, 3] in coord. frame of cam0
            normals: ground plane normal vectors of shape [B, 3]
            plane_heights: ground plane height in coordinate frame of cam0 of shape [B]
            scenes: List of scenes of shape [B]
            indices: List of frame indices of shape [B]
            cams: List of cameras of shape [B]
            cam_i: Camera index
            suffix: Metric suffix string
        Returns:
            metric: Metric of each batch of the form {'metric_name': [B, M]} where M is the # of metrics
        """
        pixel_height = (points @ normals[..., None])[..., 0]
        bad_pixels = torch.greater(plane_heights[..., None] - self.height_threshold, pixel_height).float()
        ratio = bad_pixels.mean(dim=1)
        ratio_threshold = self.ratio_threshold[cam_i] if is_list(self.ratio_threshold) else self.ratio_threshold
        bad_frames = torch.greater(ratio, ratio_threshold)
        metric = {
            f'height|below_{-self.height_threshold}m_{suffix}': torch.cat([ratio, bad_frames.float()], dim=-1)
        }
        for bad_frame, scene, idx, cam in zip(bad_frames, scenes, indices, cams):
            if not bad_frame:
                self.register_frame(cam, scene, int(idx))
        return metric

    def evaluate(self, batch: Dict, output: Dict, *args, **kwargs) -> Tuple[Dict, Dict]:
        """ Evaluate predictions
        Args:
            batch : Dictionary containing ground-truth information
            output : Dictionary containing predictions
        Returns:
            metrics : Dictionary with calculated metrics
            predictions : Dictionary with additional predictions
        """
        metrics, predictions = {}, {}
        # For each output item
        for key, val in output.items():
            # If it corresponds to this task
            if key.startswith('depth') and 'debug' not in key:
                # Loop over every context
                val = val if is_dict(val) else {0: val}
                for ctx in val.keys():
                    depth = val[ctx][0]     # only full scale

                    points = self.get_points(
                        depth=depth[:, 0],
                        intrinsics=batch['intrinsics'][ctx][:, 0],
                    )

                    # Retrieve ground plane with ransac
                    normals, plane_heights = [], []
                    for sample_points in points.cpu().numpy():
                        plane = pyrsc.Plane()
                        best_eq, best_inliers = plane.fit(sample_points, 0.01)
                        best_eq = [-float(a) for a in best_eq] if best_eq[1] > 0 else [float(a) for a in best_eq]
                        normals.append(best_eq[0:3])
                        plane_heights.append(-best_eq[3])
                    normals = torch.tensor(np.stack(normals), dtype=points.dtype, device=points.device)
                    plane_heights = torch.tensor(plane_heights, dtype=points.dtype, device=points.device)

                    metrics.update(self.eval_points(
                        points=points,
                        normals=normals,
                        plane_heights=plane_heights,
                        scenes=batch['scene'],
                        indices=batch['frame_idx'][0][0],
                        cams=batch['cam'][0],
                        cam_i=0,
                        suffix=f'({ctx}_0)_0'
                    ))

                    # Use ground-plane from first camera (typically front-facing) and apply to other cameras
                    for j, filename in enumerate(batch['filename'][ctx][1:]):
                        cam = j + 1
                        points = self.get_points(
                            depth=depth[:, cam],
                            intrinsics=batch['intrinsics'][ctx][:, cam],
                            pose=batch['pose'][ctx][:, 0].inverse() @ batch['pose'][ctx][:, cam]
                        )
                        metrics.update(self.eval_points(
                            points=points,
                            normals=normals,
                            plane_heights=plane_heights,
                            scenes=batch['scene'],
                            indices=batch['frame_idx'][0][cam],
                            cams=batch['cam'][cam],
                            cam_i=cam,
                            suffix=f'({ctx}_{j})_0'
                        ))
        return dict_remove_nones(metrics), {}
