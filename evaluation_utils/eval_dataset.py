import os
import csv
import torch
from tqdm import tqdm
from lietorch import SE3

from r3d3.r3d3 import R3D3
from r3d3.modules.completion import DepthCompletion

from vidar.utils.write import write_npz
from vidar.utils.config import read_config, get_folder_name, load_class, Config, cfg_has
from vidar.utils.networks import load_checkpoint
from vidar.utils.config import recursive_assignment

from evaluation_utils.dataloader_wrapper import setup_dataloaders, SceneIterator, SampleIterator
from vidar.utils.setup import setup_metrics
from r3d3.utils import pose_matrix_to_quaternion


def load_completion_network(cfg: Config) -> DepthCompletion:
    """ Loads completion network with vidar framework
    Args:
        cfg: Completion network config
    Returns:
        Completion network with loaded checkpoint if path is provided
    """
    folder, name = get_folder_name(cfg.file, 'networks', root=cfg_has(cfg, 'root', 'vidar/arch'))
    network = load_class(name, folder)(cfg)
    recursive_assignment(network, cfg, 'networks', verbose=True)
    if cfg_has(cfg, 'checkpoint'):
        network = load_checkpoint(
            network,
            cfg.checkpoint,
            strict=False,
            verbose=True,
            prefix='completion'
        )
    return network.networks.cuda().eval()


class Evaluator:
    """ R3D3 evaluation module
    """
    def __init__(self, args):
        """
        Args:
            args: Arguments from argparser containing
                config: Path to vidar-config file (yaml) containing configurations for dataset, metrics (optional) and
                    completion network (optional)
                R3D3-args: As described by R3D3
                training_data_path: Path to directory where R3D3 training samples should be stored. If None, training
                    samples are not stored. Default - None
                prediction_data_path: Path to directory where R3D3 predictions should be stored. If None, predictions
                    are not stored. Default - None
        """
        self.args = args
        self.cfg = read_config(self.args.config)
        self.dataloaders = setup_dataloaders(self.cfg.datasets, n_workers=args.n_workers)
        self.completion_network = None
        if cfg_has(self.cfg, 'networks') and cfg_has(self.cfg.networks, 'completion'):
            self.completion_network = load_completion_network(
                self.cfg.networks.completion
            )
        self.metrics = {}
        if cfg_has(self.cfg, 'evaluation'):
            self.metrics = setup_metrics(self.cfg.evaluation)
        self.depth_results = []
        self.trajectory_results = []
        self.confidence_stats = []

        self.training_data_path = args.training_data_path
        self.prediction_data_path = args.prediction_data_path

    def eval_scene(self, scene: str, n_cams: int, sample_iterator: SampleIterator) -> None:
        """ Evaluates a given scene by 1. Initializing R3D3, 2. Running R3D3 for each frame, 3. Terminate R3D3
        Args:
            scene: Current scene to be processed
            n_cams: Number of cameras
            sample_iterator: Iterator yielding samples from each timestep in chronological order
        """
        scene_depth_results = []
        pred_poses, gt_poses = [], []
        depth_res_idx = 0
        pred_pose_list = []
        pose_keys = ['x', 'y', 'z', 'r', 'i', 'j', 'k']

        r3d3 = R3D3(
            completion_net=self.completion_network,
            n_cams=n_cams,
            **{key.replace("r3d3_", ""): val for key, val in vars(self.args).items() if key.startswith("r3d3_")}
        )

        for timestamp, sample in enumerate(tqdm(sample_iterator, desc='Sample', position=0, leave=True)):
            pose = SE3(pose_matrix_to_quaternion(sample['pose'][0][0]).cuda())
            pose = pose.inv()
            pose_rel = (pose * pose[0:1].inv())

            intrinsics = sample['intrinsics'][0][0, :, [0, 1, 0, 1], [0, 1, 2, 2]]
            is_keyframe = 'depth' in sample and sample['depth'][0].max() > 0.

            output = r3d3.track(
                tstamp=float(timestamp),
                image=(sample['rgb'][0][0] * 255).type(torch.uint8).cuda(),
                intrinsics=intrinsics.cuda(),
                mask=(sample['mask'][0][0, :, 0] > 0).cuda() if 'mask' in sample else None,
                pose_rel=pose_rel.data
            )

            output = {key: data.cpu() if torch.is_tensor(data) else data for key, data in output.items()}
            pred_pose = None
            if output['pose'] is not None:
                pred_pose = (pose_rel.cpu() * SE3(output['pose'][None])).inv()
                pred_pose_list.append(
                    {'filename': sample['filename'][0][0][0], **dict(zip(pose_keys, pred_pose[0].data.numpy()))}
                )
                pred_poses.append(pred_pose.matrix())
                gt_poses.append(sample['pose'][0][0])
            if output['disp_up'] is not None and 'depth' in self.metrics and is_keyframe:
                results = {
                    'ds_idx': sample['idx'][0],
                    'sc_idx': torch.tensor(depth_res_idx, dtype=sample['idx'][0].dtype, device=sample['idx'][0].device),
                    'scene': scene
                }
                results.update({key: metric[0] for key, metric in self.metrics['depth'].evaluate(
                    batch=sample,
                    output={'depth': {0: [1 / output['disp_up'].unsqueeze(0).unsqueeze(2)]}}
                )[0].items()})
                scene_depth_results.append(results)
                depth_res_idx += 1

            if self.training_data_path is not None and pred_pose is not None:
                for cam, filename in enumerate(sample['filename'][0]):
                    write_npz(
                        os.path.join(
                            self.training_data_path,
                            filename[0].replace('rgb', 'r3d3').replace('CAM_', 'R3D3_') + '.npz'
                        ),
                        {
                            'intrinsics': intrinsics[cam].numpy(),
                            'pose': pred_pose[cam].data.numpy(),
                            'disp': output['disp'][cam].numpy()[None],
                            'disp_up': output['disp_up'][cam].numpy()[None],
                            'conf': output['conf'][cam].numpy()[None],
                        }
                    )
            if self.prediction_data_path is not None and output['disp_up'] is not None and is_keyframe:
                for cam, filename in enumerate(sample['filename'][0]):
                    write_npz(
                        os.path.join(
                            self.prediction_data_path,
                            filename[0] + '_depth(0)_pred.npz'
                        ),
                        {
                            'depth': (1.0 / output['disp_up'][cam].numpy()),
                            'intrinsics': intrinsics[cam].numpy(),
                            'd_info': 'r3d3_depth',
                            't': float(timestamp)
                        }
                    )

        # Terminate
        del r3d3
        torch.cuda.empty_cache()

        if self.prediction_data_path is not None and len(pred_pose_list) > 0:
            pose_dir = os.path.join(self.prediction_data_path, 'poses')
            if not os.path.exists(pose_dir):
                os.makedirs(pose_dir)
            with open(os.path.join(pose_dir, f'{scene}_poses.csv'), 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=pred_pose_list[0].keys())
                writer.writeheader()
                writer.writerows(pred_pose_list)
        self.depth_results.extend([{'idx': res['ds_idx'], **res} for res in scene_depth_results])
        if 'depth' in self.metrics and len(scene_depth_results) >= 1:
            reduced_data = self.metrics['depth'].reduce_metrics(
                [[{'idx': res['sc_idx'], **res} for res in scene_depth_results]],
                [scene_depth_results], strict=False
            )
            self.metrics['depth'].print(reduced_data, [f'scene-{scene}'])
        if 'trajectory' in self.metrics and len(gt_poses) >= 2:
            results = {'scene': scene}
            results.update(self.metrics['trajectory'].evaluate(
                batch={'trajectory': {0: torch.stack(gt_poses)}},
                output={'trajectory': {0: [torch.stack(pred_poses)]}}
            )[0])
            self.trajectory_results.append({'idx': torch.tensor(len(self.trajectory_results)), **results})
            reduced_data = self.metrics['trajectory'].reduce_metrics(
                [[{'idx': torch.tensor(0), **results}]],
                [[results]], strict=False
            )
            self.metrics['trajectory'].print(reduced_data, [f'scene-{scene}'])

    def eval_datasets(self) -> None:
        """ Evaluates datasets consisting of multiple scenes
        """
        for dataloader in tqdm(self.dataloaders, desc='Datasets', position=2, leave=True):
            n_cams = len(dataloader.dataset.cameras)
            pbar = tqdm(SceneIterator(dataloader), desc='Scenes', position=1, leave=True)
            for scene, sample_iterator in pbar:
                pbar.set_postfix_str("Processing Scene - {}".format(scene))
                pbar.refresh()
                self.eval_scene(scene, n_cams, sample_iterator)

            if 'depth' in self.metrics and len(self.depth_results) > 0:
                reduced_data = self.metrics['depth'].reduce_metrics(
                    [self.depth_results],
                    [dataloader.dataset], strict=False
                )
                self.metrics['depth'].print(reduced_data, ['Overall'])
            if 'trajectory' in self.metrics and len(self.trajectory_results) > 0:
                reduced_data = self.metrics['trajectory'].reduce_metrics(
                    [self.trajectory_results],
                    [self.trajectory_results], strict=False
                )
                self.metrics['trajectory'].print(reduced_data, ['Overall'])
            # Use to evaluate confidence statistics => Can find scenes where metric was not recovered / failed
            # if len(self.confidence_stats) > 0:
            #     confidence_stats_summary = {}
            #     for element in self.confidence_stats:
            #         scene = element['scene']
            #         if scene not in confidence_stats_summary:
            #             stats_keys = [key for key in element.keys() if key is not scene]
            #             confidence_stats_summary[scene] = {k: [] for k in stats_keys if k not in ['scene', 'idx']}
            #         for key in confidence_stats_summary[scene]:
            #             confidence_stats_summary[scene][key].append(element[key])
            #     confidence_stats_summary = {
            #         scene: {key: sum(val) / len(val) for key, val in stats.items()}
            #         for scene, stats in confidence_stats_summary.items()
            #     }
            #     import csv
            #     with open('confidence_stats.csv', 'w', newline='') as output_file:
            #         dict_writer = csv.DictWriter(output_file, list(self.confidence_stats[0].keys()))
            #         dict_writer.writeheader()
            #         dict_writer.writerows(self.confidence_stats)
