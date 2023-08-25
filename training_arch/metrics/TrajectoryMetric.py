import numpy as np
import torch
from typing import Dict, List, Tuple, Optional

from vidar.metrics.base import BaseEvaluation
from vidar.utils.config import Config, cfg_has
from vidar.utils.data import dict_remove_nones
from vidar.utils.distributed import on_rank_0
from vidar.utils.logging import pcolor

from evo.core.trajectory import PoseTrajectory3D
from evo.core import sync
import evo.main_ape as main_ape
from evo.core.metrics import PoseRelation

from r3d3.utils import pose_matrix_to_quaternion


class TrajectoryMetric(BaseEvaluation):
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
            name='trajectory',
            task='trajectory',
            metrics=('ATE [m]',),
        )
        self.eval_cam = cfg_has(cfg, 'eval_cam', 0)

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
            if key.startswith('trajectory') and 'debug' not in key:
                # Loop over every context
                traj_gt = pose_matrix_to_quaternion(batch['trajectory'][0][:, self.eval_cam]).numpy()
                traj_est = pose_matrix_to_quaternion(val[0][0][:, self.eval_cam]).numpy()
                timestamps = np.arange(len(traj_gt), dtype=traj_gt.dtype)
                traj_est_evo = PoseTrajectory3D(
                    positions_xyz=traj_est[:, :3],
                    orientations_quat_wxyz=traj_est[:, 3:],
                    timestamps=timestamps
                )
                traj_gt_evo = PoseTrajectory3D(
                    positions_xyz=traj_gt[:, :3],
                    orientations_quat_wxyz=traj_gt[:, 3:],
                    timestamps=timestamps
                )
                traj_est_evo_sync, traj_gt_evo_sync = sync.associate_trajectories(traj_est_evo, traj_gt_evo)
                ape_res = main_ape.ape(
                    traj_gt_evo_sync,
                    traj_est_evo_sync,
                    est_name='traj',
                    pose_relation=PoseRelation.translation_part,
                    align=False,
                    correct_scale=False
                )
                ape_res_scaled = main_ape.ape(
                    traj_gt_evo_sync,
                    traj_est_evo_sync,
                    est_name='traj',
                    pose_relation=PoseRelation.translation_part,
                    align=True,
                    correct_scale=True
                )
                metrics = {
                    f'{key}|unscaled_{({self.eval_cam})}': torch.tensor(ape_res.stats['rmse'], dtype=torch.float32),
                    f'{key}|scaled_{({self.eval_cam})}': torch.tensor(ape_res_scaled.stats['rmse'], dtype=torch.float32),
                }
        return dict_remove_nones(metrics), {}
