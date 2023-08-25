from typing import Dict, Tuple

import torch
from vidar.arch.networks.BaseNet import BaseNet
from vidar.utils.config import Config, cfg_has

from r3d3.modules.completion import DepthCompletion


class CompletionNet(BaseNet):
    """ Vidar wrapper for R3D3 depth completion network
    """
    def __init__(self, cfg: Config):
        super(CompletionNet, self).__init__(cfg)
        self.networks = DepthCompletion(
            input_disp_mode=cfg_has(cfg, 'inp_disp_mode', 'normalize'),
            min_conf=cfg_has(cfg, 'min_conf', 0.0),
            disp_dist=cfg_has(cfg, 'disp_dist', None),
            disp_up_dist=cfg_has(cfg, 'disp_up_dist', None),
            min_depth_in=cfg_has(cfg, 'min_depth_in', 1.0),
            max_depth_in=cfg_has(cfg, 'max_depth_in', 200.0),
            min_depth_out=cfg_has(cfg, 'min_depth_out', 1.0),
            max_depth_out=cfg_has(cfg, 'max_depth_out', 200.0),
            focal_norm=cfg_has(cfg, 'focal_norm', 1.0),
            mask_disp_clip=cfg_has(cfg, 'mask_disp_clip', False)
        )

    def train(self, mode: bool = True):
        self.networks.train(mode=mode)

    def forward(
        self,
        image: torch.Tensor,
        disps: torch.Tensor,
        conf: torch.Tensor,
        disps_up: torch.Tensor,
        conf_up: torch.Tensor,
        focal_length: torch.Tensor
    ) -> Dict[Tuple, torch.Tensor]:
        return self.networks(image, disps, conf, disps_up, conf_up, focal_length)
