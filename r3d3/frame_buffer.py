from typing import Tuple, Optional, Union
from enum import Enum

import torch
from torchvision.transforms import Resize, InterpolationMode
from lietorch import SE3

from r3d3.r3d3_net import cvx_upsample
from r3d3.modules.completion import DepthCompletion
import r3d3.geom.projective_ops as pops
import r3d3_backends


class CompletionMode(Enum):
    NONE = 0        # No completion
    CURRENT = 1     # Complete only current timestep
    OPT_WINDOW = 2  # Complete all frames in bundle-adjustment pose optimization window
    ALL = 3         # Complete all frames (including such for which pose was fixed during optimization)


class FrameBuffer:
    """ Class for storing frames, poses and depth as well as applying operations on such (completion & DBA)"""
    def __init__(
            self,
            image_size: Tuple[int, int],
            buffer: int,
            n_cams: int,
            scale: Optional[float] = 15.0,
            completion_net: Optional[DepthCompletion] = None,
            completion_mode: CompletionMode = CompletionMode.NONE,
    ):
        """
        Args:
            image_size: Image size,
            buffer_size: Size of R3D3 frame buffer (=# timesteps). Default: 0 - buffer size depending on ref-window
            n_cams: Number of cameras
            scale: Scale at which to operate
            completion_net: Completion network instance
            completion_mode: Which frames should be completed
        """
        self.counter = 0
        self.height = ht = image_size[0]
        self.width = wd = image_size[1]
        self.n_cams = n_cams
        self.scale = scale

        self.completion_net = completion_net
        self.completion_mode = completion_mode

        # state attributes
        self.timestamp = torch.zeros(buffer, device="cuda", dtype=torch.float)
        self.images = torch.zeros(buffer, n_cams, 3, ht, wd, device="cuda", dtype=torch.uint8)
        self.dirty = torch.zeros(buffer, device="cuda", dtype=torch.bool)
        self.poses = torch.zeros(buffer, 7, device="cuda", dtype=torch.float)
        self.rel_poses = torch.zeros(buffer, n_cams, 7, device="cuda", dtype=torch.float)
        self.disps = torch.ones(buffer, n_cams, ht // 8, wd // 8, device="cuda", dtype=torch.float)
        self.disps_up = torch.zeros(buffer, n_cams, ht, wd, device="cuda", dtype=torch.float)
        self.masks = torch.ones(n_cams, ht // 8, wd // 8, device="cuda", dtype=torch.bool)
        self.intrinsics = torch.zeros(buffer, n_cams, 4, device="cuda", dtype=torch.float)

        # feature attributes
        self.corr_feat = torch.zeros(buffer, n_cams, 1, 128, ht // 8, wd // 8,
                                     dtype=torch.half, device="cuda")
        self.hidden = torch.zeros(buffer, n_cams, 128, ht // 8, wd // 8,
                                  dtype=torch.half, device="cuda")
        self.ctx_feat = torch.zeros(buffer, n_cams, 128, ht // 8, wd // 8,
                                    dtype=torch.half, device="cuda")

        # initialize poses to identity transformation
        self.poses[:] = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device="cuda")

        self.buffer = buffer

    @property
    def images_flat(self) -> torch.Tensor:
        return self.images.view(-1, *self.images.shape[2:])

    @property
    def dirty_flat(self) -> torch.Tensor:
        return self.dirty.unsqueeze(1).expand(-1, self.n_cams).reshape(-1)

    @property
    def disps_flat(self) -> torch.Tensor:
        return self.disps.view(-1, *self.disps.shape[2:])

    @property
    def disps_up_flat(self) -> torch.Tensor:
        return self.disps_up.view(-1, *self.disps_up.shape[2:])

    @property
    def intrinsics_flat(self) -> torch.Tensor:
        return self.intrinsics.view(-1, *self.intrinsics.shape[2:])

    @property
    def rel_poses_flat(self) -> torch.Tensor:
        return self.rel_poses.view(-1, *self.rel_poses.shape[2:])

    @property
    def corr_feat_flat(self) -> torch.Tensor:
        return self.corr_feat.view(-1, *self.corr_feat.shape[-3:])

    def get_abs_poses(self) -> torch.Tensor:
        """ Returns absolute poses as world2cam
        Returns:
            Absolute poses of shape (buffer, n_cams, 7)
        """
        return (SE3(self.rel_poses) * SE3(self.poses.unsqueeze(1))).data

    def append(
            self,
            timestamp: float,
            image: torch.Tensor,
            pose_rel: torch.Tensor,
            intrinsics: torch.Tensor,
            corr_feat: torch.Tensor,
            hidden_init: torch.Tensor,
            ctx_feat: torch.Tensor,
            pose_init: Optional[torch.Tensor] = None,
            disp_init: Optional[Union[torch.Tensor, float]] = None,
            mask: Optional[torch.Tensor] = None,
    ):
        """ Appends frames from timestep to frame_buffer
        With H, W the full image height and width and C the number of cameras ...
        Args:
            timestamp: Timesamp of current frames
            image: uint8 images of shape (C, 3, H, W) in [0, 255]
            pose_rel: Relative poses which are ref-point2camx of shape (C, 7)
            intrinsics: Intrinsics of low resolution image (H/8, W/8) of shape (C, 4)
            corr_feat: Correlation features of shape (C, 1, 128, H/8, W/8)
            hidden_init: Hidden initialization of GRU of shape (C, 128, H/8, W/8)
            ctx_feat: Context features of shape (C, 128, H/8, W/8)
            pose_init: Initial pose of shape (7) and the form [x, y, z, quat] which is world2ref-point
            disp_init: Initial disparity, either of shape (C, H/8, W/8) or a single value (uniform initialization)
            mask: Self-occlusion boolean mask where valid regions are True of shape (C, H/8, W/8)
        """
        self.timestamp[self.counter] = timestamp
        self.images[self.counter] = image
        self.rel_poses[self.counter] = pose_rel

        self.intrinsics[self.counter] = intrinsics
        self.corr_feat[self.counter] = corr_feat
        self.hidden[self.counter] = hidden_init
        self.ctx_feat[self.counter] = ctx_feat
        if pose_init is not None:
            self.poses[self.counter] = pose_init
        if disp_init is not None:
            self.disps[self.counter] = disp_init
        if mask is not None:
            mask_down = Resize(
                size=(self.height // 8, self.width // 8),
                interpolation=InterpolationMode.NEAREST
            )(mask)
            self.masks[:] = mask_down
        self.counter += 1

    @staticmethod
    def format_indicies(ii: torch.Tensor, jj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ to device, long, {-1} """
        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii)
        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj)

        ii = ii.to(device="cuda", dtype=torch.long).reshape(-1)
        jj = jj.to(device="cuda", dtype=torch.long).reshape(-1)

        return ii, jj

    @torch.cuda.amp.autocast(enabled=False)
    def upsample(self, ix: torch.Tensor, mask: torch.Tensor) -> None:
        """ upsample disparity
        Args:
            ix: Indices of disps which should be upsampled of shape (N, 2) of the form [[i, c], ...] where
                i in {0, ..., buffer-length-1} and c in {0, ..., n-cameras-1} of type long
            mask: Upsampling mask
        """
        disps_up = cvx_upsample(self.disps[ix[:, 0], ix[:, 1]].unsqueeze(-1), mask)
        self.disps_up[ix[:, 0], ix[:, 1]] = disps_up.squeeze()

    def mono_from_input(
            self,
            image: torch.Tensor,
            intrinsics: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """ Uses completion network to make monocular prediction by setting input depth + conf to 0. Returns None
        if completion net is not available.
        Args:
            image: Input image of shape (C, 3, H, W)
            intrinsics: Camera intrinsics of shape (C, 4) and with format [[fx, fy, cx, cy], ...]
        Returns:
            disp_up: Disparity (1 / depth) at original scale of shape (C, H, W)
            disp: Disparity (1 / depth) at reduced scale of shape (C, H/8, W/8)
        """
        if self.completion_net is not None:
            output = self.completion_net(
                image,
                torch.zeros_like(self.disps[0])[:, None],
                torch.zeros_like(self.disps[0])[:, None] - 1,
                torch.zeros_like(self.disps_up[0])[:, None],
                torch.zeros_like(self.disps_up[0])[:, None] - 1,
                intrinsics
            )
            return output[('disp', 0)][:, 0], output[('disp', 3)][:, 0]
        else:
            return None, None

    def mono_from_buffer(
            self,
            t: float,
    ) -> None:
        """ Uses completion network to make monocular prediction by setting input depth + conf to 0. Uses data from
        frame-buffer. Writes result back into frame buffer
        Args:
            t: Current time-step
        """
        if self.completion_net is not None:
            idx = t - 1
            images = self.images[idx] / 255.
            intrinsics = self.intrinsics[idx, :, 0] * 8
            disp_up, disp = self.mono_from_input(images, intrinsics)
            if disp_up is not None and disp is not None:
                self.disps_up[idx] = disp_up * self.scale
                if self.disps[idx].min() == self.disps[idx].max():  # do not overwrite existing initialization
                    self.disps[idx] = disp * self.scale

    def complete(
            self,
            t: float,
            weight: torch.Tensor,
            cii: torch.Tensor,
            t_from_ref: Optional[float] = None,
            t_from_opt: Optional[float] = None,
    ) -> None:
        """ Completes geometric depth with completion network. Nothing happens if either completion_mode is
        CompletionMode.NONE or no completion_net is set.
        Args:
            t: Current time-step
            weight: Flow confidence of shape (|E|, 2, H/8, W/8)
            cii: Outgoing nodes of edges in covisibility graph of the shape (|E|, 2) with |E| = # edges of the form
                [[i, c], ...] where i in {0, ..., buffer-length-1} and c in {0, ..., n-cameras-1} of type long
            t_from_ref: time-step where reference window starts (t_from_ref <= t_from_opt <= t)
            t_from_opt: time-step where optimization window starts (t_from_ref <= t_from_opt <= t)
        """
        if (self.completion_mode != CompletionMode.NONE) and self.completion_net is not None:
            # Completion prediction (input depth / conf = 0)
            t_from_ref = t if t_from_ref is None else t_from_ref
            t_from_opt = t if t_from_opt is None else t_from_opt

            t_from = t
            if self.completion_mode == CompletionMode.ALL:
                t_from = t_from_ref
            elif self.completion_mode == CompletionMode.OPT_WINDOW:
                t_from = t_from_opt

            idx_range = torch.arange(t_from - 1, t, device=cii.device)
            weight = torch.stack([weight[i] for i in range(len(cii)) if cii[i, 0] in idx_range])
            cii = torch.stack([ci for ci in cii if ci[0] in idx_range])
            ix = torch.unique(cii, dim=0)

            image = self.images[ix[:, 0], ix[:, 1]] / 255.
            disps = self.disps[ix[:, 0], ix[:, 1]] / self.scale
            disps_up = self.disps_up[ix[:, 0], ix[:, 1]] / self.scale
            conf = torch.cat([weight.mean(-3)[(cii == i).all(-1), :, :].max(dim=0, keepdim=True)[0] for i in ix], dim=0)
            # conf = torch.cat([weight.mean(-3)[(cii==i).all(-1), :, :].mean(dim=0, keepdim=True) for i in ix], dim=0)
            conf_up = Resize(size=(self.height, self.width))(conf)
            intr = self.intrinsics[ix[:, 0], ix[:, 1], 0] * 8

            output = self.completion_net(
                image, disps[:, None], conf[:, None], disps_up[:, None], conf_up[:, None], intr
            )
            self.disps_up[ix[:, 0], ix[:, 1]] = output[('disp', 0)][:, 0] * self.scale
            self.disps[ix[:, 0], ix[:, 1]] = output[('disp', 3)][:, 0] * self.scale

    def reproject(self, cii: torch.Tensor, cjj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Project image coordinates from nodes cii to cjj
        Args:
            cii: Outgoing nodes of shape (|E|, 2)
            cjj: Incoming nodes of shape (|E|, 2)
        Returns:
            coords: Projected coordinates of shape (1, |E|, H/8, W/8, 2)
            valid_mask: True if projection inside image, of shape (1, |E|, H/8, W/8)
        """
        ii = self.n_cams * cii[:, 0] + cii[:, 1]  # flatten graph
        jj = self.n_cams * cjj[:, 0] + cjj[:, 1]  # flatten graph
        ii, jj = FrameBuffer.format_indicies(ii, jj)
        poses_w2cam = (SE3(self.rel_poses) * SE3(self.poses[:, None])) # .view((1, -1))

        # old implementation
        coords, valid_mask = pops.projective_transform(
            poses_w2cam.view((1, -1)),
            self.disps_flat[None],
            self.intrinsics_flat[None],
            ii, jj
        )
        return coords, valid_mask

        # new implementation
        # coords, valid_mask = r3d3_backends.projmap(
        #     poses_w2cam.data.view(-1, 7),
        #     self.disps_flat,
        #     self.intrinsics_flat,
        #     ii, jj
        # )
        # return coords[None], valid_mask[None]

    def distance(
            self,
            cii: Optional[torch.Tensor] = None,
            cjj: Optional[torch.Tensor] = None,
            bidirectional: Optional[bool] = True
    ):
        """ Calculates distance between frames via mean induced flow . If cii & cjj are not given, distance is
        calculated between all pairs.
        Args:
            cii: Outgoing nodes of shape (|E|, 2)
            cjj: Incoming nodes of shape (|E|, 2)
            bidirectional: If distance should be calculated as (mean-flow(i -> j) + mean-flow(j -> i)) / 2
        """
        n_nodes = None
        if cii is None or cjj is None:
            n_nodes = self.counter * self.n_cams
            ii, jj = torch.meshgrid(torch.arange(n_nodes), torch.arange(n_nodes))
        else:
            ii = self.n_cams * cii[:, 0] + cii[:, 1]
            jj = self.n_cams * cjj[:, 0] + cjj[:, 1]

        ii, jj = FrameBuffer.format_indicies(ii, jj)

        abs_poses = self.get_abs_poses().view(-1, 7)
        if bidirectional:
            d1 = r3d3_backends.frame_distance(abs_poses, self.disps_flat, self.intrinsics_flat, ii, jj, 0.6)
            d2 = r3d3_backends.frame_distance(abs_poses, self.disps_flat, self.intrinsics_flat, jj, ii, 0.6)
            distance = .5 * (d1 + d2)
        else:
            distance = r3d3_backends.frame_distance(abs_poses, self.disps_flat, self.intrinsics_flat, ii, jj, 0.6)
        if n_nodes is not None:
            return distance.reshape(n_nodes, n_nodes)

        return distance

    def ba(
            self,
            target: torch.Tensor,
            weight: torch.Tensor,
            eta: torch.Tensor,
            ii: torch.Tensor,
            jj: torch.Tensor,
            cii: torch.Tensor,
            cjj: torch.Tensor,
            t0: float,
            t1: float,
            itrs: Optional[int] = 2,
            motion_only: Optional[bool] = False
    ) -> None:
        """ dense bundle adjustment (DBA)
            [t0, t1] window of bundle adjustment optimization
        Args:
            target: Predicted reprojected coordinates from node ci -> cj of shape (|E|, 2, H/8, W/8)
            weight: Predicted confidence weights of "target" of shape (|E|,2, H/8, W/8)
            eta: Levenberg-marquart damping factor on depth of shape (|V|, H/8, W/8)
            ii: Timesteps of outgoing edges of shape (|E|)
            jj: Timesteps of incoming edges of shape (|E|)
            cii: Nodes of outgoing edges of shape (|E|, 2)
            cjj: Nodes of incoming edges of shape (|E|, 2)
            t0: Optimization window start time
            t1: Optimitation window end time
            itrs: Number of BA iterations
            motion_only: True: Only pose is optimized
        """
        r3d3_backends.ba(
            self.poses,
            self.rel_poses_flat,
            self.disps_flat,
            self.intrinsics_flat,
            target,
            weight,
            eta,
            ii.contiguous(), jj.contiguous(),
            cii, cjj,
            t0, t1,
            itrs, 1e-4, 0.1, motion_only
        )
        self.disps.clamp_(min=0.001)


