from typing import Union, Optional, List, Dict, Tuple
from enum import Enum
import torch
import numpy as np

from lietorch import SE3

from r3d3.frame_buffer import FrameBuffer
from r3d3.modules.corr import CorrBlock, CorrBlockNew
from r3d3.geom.graph_utils import get_frustum_intersections
import r3d3.geom.projective_ops as pops
import r3d3_backends


Device = Union[torch.device, str, None]
CamNodeList = torch.Tensor  # List of nodes with shape (|V|, 2) the form [[time_i, cam_i], ...]
NodeList = torch.Tensor     # Flattened list of nodes with shape (|V|, 2) the form [N_cams * time_i + cam_i, ...]


class CorrelationMode(Enum):
    VOLUME = "volume"       # Calculate correlation volume for each edge when edge is created
    LOWMEM = "lowmem"       # Calculate correlated features directly given a feature matching


class CovisGraph:
    def __init__(
            self,
            frame_buffer: FrameBuffer,
            update_op: torch.nn.Module,
            n_edges_max: Optional[int] = -1,
            device: Optional[Device] = "cuda:0",
            corr_impl: Optional[CorrelationMode] = CorrelationMode.VOLUME,
            disable_comp_inter_flow: Optional[bool] = False,
    ):
        """ Class for storing and modifying covisibility graph
        Args:
            frame_buffer: Buffer of frames etc.
            update_op: Flow estimation module
            n_edges_max: Maximum number of edges to store
            device: Device on which to store graph
            corr_impl: Correlation volume implementation type
            disable_comp_inter_flow: If compensation of inter-cam rotation should be disabled
        """
        self.frame_buffer = frame_buffer
        self.update_op = update_op
        self.device = device
        self.n_edges_max = n_edges_max
        self.corr_impl = corr_impl
        self.disable_comp_inter_flow = disable_comp_inter_flow

        # operator at 1/8 resolution
        self.ht = ht = frame_buffer.height // 8
        self.wd = wd = frame_buffer.width // 8

        self.coords0 = pops.coords_grid(ht, wd, device=device)
        self.cii = torch.as_tensor([], dtype=torch.long, device=device)
        self.cjj = torch.as_tensor([], dtype=torch.long, device=device)
        self.age = torch.as_tensor([], dtype=torch.long, device=device)

        # ToDo: Check if new implementation does not worsen results
        # old implementation
        # self.corr = None
        # new implementation
        self.corr = CorrBlockNew(self.n_edges_max)

        self.hidden, self.ctx_feat = None, None
        self.damping = 1e-6 * torch.ones_like(self.frame_buffer.disps)

        self.target = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)
        self.weight = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)

        self.neighborhood = None
        self.neigh_rot_reprojection = None

    @property
    def cii_flat(self) -> NodeList:
        return self.flatten_nodes(self.cii)

    @property
    def cjj_flat(self) -> NodeList:
        return self.flatten_nodes(self.cjj)

    def flatten_nodes(self, nodes: CamNodeList) -> NodeList:
        return self.frame_buffer.n_cams * nodes[:, 0] + nodes[:, 1]

    def __filter_repeated_edges(self, cii: CamNodeList, cjj: CamNodeList) -> Tuple[CamNodeList, CamNodeList]:
        """ Remove duplicate edges """

        keep = torch.zeros(cii.shape[0], dtype=torch.bool, device=cii.device)
        eset = set([(ci[0].item(), ci[1].item(), cj[0].item(), cj[1].item()) for ci, cj in zip(self.cii, self.cjj)])

        for k, (ci, cj) in enumerate(zip(cii, cjj)):
            keep[k] = (ci[0].item(), ci[1].item(), cj[0].item(), cj[1].item()) not in eset

        return cii[keep], cjj[keep]

    def filter_edges(self) -> None:
        """ Remove bad edges """
        conf = torch.mean(self.weight, dim=[0, 2, 3, 4])
        mask = (torch.abs(self.cii[:, 0] - self.cjj[:, 0]) > 2) & (conf < 0.001)

        self.rm_factors(mask, store=False)

    def clear_edges(self) -> None:
        """ Remove all edges """
        self.rm_factors(self.cii[:, 0] >= 0)
        self.hidden = None
        self.ctx_feat = None

    @staticmethod
    def index(data: torch.Tensor, idx: CamNodeList) -> torch.Tensor:
        """ Index data with cam-nodes
        Args:
            data: Tensor of shape (N, C, ...)
            idx: Indices of shape (M, 2)
        Returns:
            Indexed data of shape (M, ...)
        """
        return data[idx[:, 0], idx[:, 1]]

    @torch.cuda.amp.autocast(enabled=True)
    def add_factors(self, cii: CamNodeList, cjj: CamNodeList, remove: Optional[bool] = False) -> None:
        """ Add edges to factor graph, calculates correlation volume if "CorrelationMode.VOLUME" is used
        Args:
            cii: Nodes i (from) of shape (N, 2) where N is the number of new edges
            cjj: Nodes j (to) of shape (N, 2) where N is the number of new edges
            remove: If nodes should be removed if existing + new nodes > total # nodes
        """

        if not isinstance(cii, torch.Tensor):
            cii = torch.as_tensor(cii, dtype=torch.long, device=self.device)

        if not isinstance(cjj, torch.Tensor):
            cjj = torch.as_tensor(cjj, dtype=torch.long, device=self.device)

        # remove duplicate edges
        cii, cjj = self.__filter_repeated_edges(cii, cjj)

        if cii.shape[0] == 0:
            return

        # place limit on number of factors
        if 0 < self.n_edges_max < self.cii.shape[0] + cii.shape[0] and self.corr is not None and remove:
            ix = torch.arange(len(self.age))[torch.argsort(self.age).cpu()]
            self.rm_factors(ix >= self.n_edges_max - cii.shape[0], store=True)

        # correlation volume for new edges
        if self.corr_impl == CorrelationMode.VOLUME:
            fmap1 = CovisGraph.index(self.frame_buffer.corr_feat, cii)[:, 0].to(self.device).unsqueeze(0)
            fmap2 = CovisGraph.index(self.frame_buffer.corr_feat, cjj)[:, 0].to(self.device).unsqueeze(0)

            # ToDo: Check if new implementation does not worsen results
            # old implementation
            # corr = CorrBlock(fmap1, fmap2)
            # self.corr = corr if self.corr is None else self.corr.cat(corr)
            # new implementation
            self.corr.add(fmap1, fmap2)

        with torch.cuda.amp.autocast(enabled=False):
            target, _ = self.frame_buffer.reproject(cii, cjj)
            weight = torch.zeros_like(target)

        self.cii = torch.cat([self.cii, cii], 0)
        self.cjj = torch.cat([self.cjj, cjj], 0)
        self.age = torch.cat([self.age, torch.zeros_like(cii[:, 0])], 0)

        self.target = torch.cat([self.target, target], 1)
        self.weight = torch.cat([self.weight, weight], 1)

        self.ctx_feat = CovisGraph.index(self.frame_buffer.ctx_feat, self.cii).to(self.device).unsqueeze(0)
        self.hidden = CovisGraph.index(self.frame_buffer.hidden, self.cii).to(self.device).unsqueeze(0)

    @torch.cuda.amp.autocast(enabled=True)
    def rm_factors(
            self,
            mask: torch.Tensor
    ) -> None:
        """ Remove masked edges from covisivility graph
        Args:
            mask: Mask tensor of shape (N) where N is the number of nodes in self.cii / self.cjj (= |V|)
        """
        self.cii = self.cii[~mask]
        self.cjj = self.cjj[~mask]
        self.age = self.age[~mask]

        if self.corr_impl == CorrelationMode.VOLUME:
            self.corr = self.corr[~mask]

        if self.hidden is not None:
            self.hidden = self.hidden[:, ~mask]

        if self.ctx_feat is not None:
            self.ctx_feat = self.ctx_feat[:, ~mask]

        self.target = self.target[:, ~mask]
        self.weight = self.weight[:, ~mask]

    def _rm_idx_from_graph(self, idx: int) -> None:
        """ Removes given timestep from covisibility graph
        Args:
            idx: Timestep to remove
        """
        mask = (self.cii[:, 0] == idx) | (self.cjj[:, 0] == idx)
        self.cii[self.cii[:, 0] >= idx, 0] -= 1
        self.cjj[self.cjj[:, 0] >= idx, 0] -= 1
        self.rm_factors(mask)

    @torch.cuda.amp.autocast(enabled=True)
    def rm_keyframe(self, ix: int) -> None:
        """ Removes timestep at index ix and copies data from index ix+1 to ix => used to remove frame from previous
        timestep.
        Args:
            ix: Index of timestep (must be current-index - 1!)
        """
        self.frame_buffer.images[ix] = self.frame_buffer.images[ix + 1]
        self.frame_buffer.poses[ix] = self.frame_buffer.poses[ix + 1]
        self.frame_buffer.disps[ix] = self.frame_buffer.disps[ix + 1]
        self.frame_buffer.intrinsics[ix] = self.frame_buffer.intrinsics[ix + 1]

        self.frame_buffer.hidden[ix] = self.frame_buffer.hidden[ix + 1]
        self.frame_buffer.ctx_feat[ix] = self.frame_buffer.ctx_feat[ix + 1]
        self.frame_buffer.corr_feat[ix] = self.frame_buffer.corr_feat[ix + 1]

        self.frame_buffer.timestamp[ix] = self.frame_buffer.timestamp[ix + 1]
        self.frame_buffer.disps_up[ix] = self.frame_buffer.disps_up[ix + 1]
        self.frame_buffer.rel_poses[ix] = self.frame_buffer.rel_poses[ix + 1]
        self._rm_idx_from_graph(idx=ix)

    @torch.cuda.amp.autocast(enabled=True)
    def rm_first_frame(self) -> None:
        self.frame_buffer.images = torch.roll(self.frame_buffer.images, shifts=-1, dims=0)
        self.frame_buffer.poses = torch.roll(self.frame_buffer.poses, shifts=-1, dims=0)
        self.frame_buffer.disps = torch.roll(self.frame_buffer.disps, shifts=-1, dims=0)
        self.frame_buffer.intrinsics = torch.roll(self.frame_buffer.intrinsics, shifts=-1, dims=0)

        self.frame_buffer.hidden = torch.roll(self.frame_buffer.hidden, shifts=-1, dims=0)
        self.frame_buffer.ctx_feat = torch.roll(self.frame_buffer.ctx_feat, shifts=-1, dims=0)
        self.frame_buffer.corr_feat = torch.roll(self.frame_buffer.corr_feat, shifts=-1, dims=0)

        self.frame_buffer.timestamp = torch.roll(self.frame_buffer.timestamp, shifts=-1, dims=0)
        self.frame_buffer.disps_up = torch.roll(self.frame_buffer.disps_up, shifts=-1, dims=0)
        self.frame_buffer.rel_poses = torch.roll(self.frame_buffer.rel_poses, shifts=-1, dims=0)
        self.frame_buffer.dirty[:-1] = True
        self._rm_idx_from_graph(idx=0)

    @torch.cuda.amp.autocast(enabled=True)
    def update(
            self,
            t0: int,
            t1: int,
            itrs: Optional[int] = 2,
            ep: Optional[float] = 1e-7,
            motion_only: Optional[bool] = False
    ) -> Tuple[torch.Tensor, torch.Tensor, CamNodeList]:
        """ Run update operator on covisibility graph
        Args:
            t0: [t0, t1] window of bundle adjustment optimization
            t1: see above
            itrs: Number of GRU + DBA iterations
            ep: Regularizer
            motion_only: If only motion should be optimized (fix depth)
        Returns:
            weight: current flow-confidence estimate of shape (|E|, 2, H, W)
            upmask: Estimated upsampling masks (1, |V|, 9 * 8 * 8, H, W)
            cii: Outgoing nodes of shape (|E|, 2)
        """

        # ToDo: Figure out if new implementation does not worsen results
        # old implementation
        rots = self.frame_buffer.rel_poses_flat.clone()
        rots[..., :3] = 0
        coords, valid_mask = pops.projective_transform(
            SE3(rots[None]),
            self.frame_buffer.disps_flat[None],
            self.frame_buffer.intrinsics_flat[None],
            self.cii_flat, self.cjj_flat
        )

        # new implementation
        # coords = self.get_rot_reprojection(self.cii, self.cjj)

        # motion features
        with torch.cuda.amp.autocast(enabled=False):
            coords1, mask = self.frame_buffer.reproject(self.cii, self.cjj)
            if self.disable_comp_inter_flow:
                motn = torch.cat([coords1 - self.coords0, self.target - coords1], dim=-1)
            else:
                motn = torch.cat([coords1 - coords, self.target - coords1], dim=-1)
            motn = motn.permute(0, 1, 4, 2, 3).clamp(-64.0, 64.0)

        corr = self.corr(coords1, self.cii_flat, self.cjj_flat)  # sample correlation features

        # GRU update operation
        self.hidden, delta, weight, damping, upmask = \
            self.update_op(self.hidden, self.ctx_feat, corr, motn, self.cii_flat, self.cjj_flat)

        with torch.cuda.amp.autocast(enabled=False):
            self.target = coords1 + delta.to(dtype=torch.float)
            self.weight = weight.to(dtype=torch.float)

            ht, wd = self.coords0.shape[0:2]
            xi = torch.unique(self.cii, dim=0)
            self.damping[xi[:, 0], xi[:, 1]] = damping

            cii, cjj, target, weight = self.cii, self.cjj, self.target, self.weight

            weight *= (self.frame_buffer.masks[cii[:, 1]]).unsqueeze(-1).unsqueeze(0)  # self-occlusion masking

            ii = self.flatten_nodes(cii)
            jj = self.flatten_nodes(cjj)

            damping = .2 * self.damping.view(-1, ht, wd)[torch.unique(ii)].contiguous() + ep
            target = target.view(-1, ht, wd, 2).permute(0, 3, 1, 2).contiguous()
            weight = weight.view(-1, ht, wd, 2).permute(0, 3, 1, 2).contiguous()

            self.frame_buffer.ba(target, weight, damping, cii[:, 0], cjj[:, 0], ii, jj, t0, t1, itrs=itrs,
                                 motion_only=motion_only)
        self.age += 1
        return weight.clone(), upmask.clone(), self.cii.clone()

    def get_neighborhood(self) -> Dict[int, List[int]]:
        """ Calculates and returns camera adjacency based on cam-frustum overlap in 2D projection on xz-plane (cam-coo.)
        Returns:
            Camera adjacency dictionary of the form {cam_i: [cam_j, ...], ...}
        """
        if self.neighborhood is None:
            intersections = get_frustum_intersections(
                poses=SE3(self.frame_buffer.rel_poses[0]).inv().matrix().cpu().numpy(),
                intrinsics=self.frame_buffer.intrinsics[0].cpu(),
                im_hw=self.frame_buffer.n_cams * [(self.ht, self.wd)],
                near=5.0 / self.frame_buffer.scale,
                far=100.0 / self.frame_buffer.scale
            )
            self.neighborhood = {}
            for cam_i in range(self.frame_buffer.n_cams):
                self.neighborhood[cam_i] = []
                for cam_j in range(self.frame_buffer.n_cams):
                    if (cam_i != cam_j) and (intersections[cam_i][cam_j] > 0.0):
                        self.neighborhood[cam_i].append(cam_j)
        return self.neighborhood

    def get_rot_reprojection(self, cii: CamNodeList, cjj: CamNodeList) -> torch.Tensor:
        """ Returns projected coordinates based on rotation between cameras
        Args:
            cii: Nodes i (from) of shape (N, 2)
            cjj: Nodes j (to) of shape (N, 2)
        Returns:
            coords: Projected coordinates of shape (N, 2, H, W) ToDo: Check
        """
        rots = self.frame_buffer.rel_poses_flat.clone()
        rots[..., :3] = 0
        coords, valid_mask = r3d3_backends.projmap(
            rots,
            self.frame_buffer.disps_flat,
            self.frame_buffer.intrinsics_flat,
            self.flatten_nodes(cii),
            self.flatten_nodes(cjj)
        )
        return coords[None]

    def get_velocity_proj(self, vel_vec: torch.Tensor) -> torch.Tensor:
        """ Projects camera vectors onto velocity vector
        Args:
            vel_vec: Current velocity vector in camera-reference frame of shape (3) - Default: [0., 0., 1.]
        Returns:
            projection of camera vector onto velocity vector (T_(cam -> ref) @ cam_vec) * vel_vec of shape (C)
        """
        vel_vec_norm = torch.cat([vel_vec, torch.ones_like(vel_vec[0:1])])
        cam_vec_norm = torch.zeros((self.frame_buffer.n_cams, 4), device=vel_vec.device, dtype=vel_vec.dtype)
        cam_vec_norm[:, 2] = self.frame_buffer.intrinsics[0, :, 0]
        cam_vec_norm[:, 3] = 1.0
        cam_2_ref = SE3(self.frame_buffer.rel_poses[0]).inv().matrix()
        return (vel_vec_norm[None] @ (cam_2_ref @ cam_vec_norm[..., None])).squeeze() - 1

    def get_temporal_edges(self, t: int, dt_intra: int, r_intra: int) -> Tuple[CamNodeList, CamNodeList]:
        """ Returns temporal edges
            t: Current time step
            dt_intra: Time-window in which temporal edges are added
            r_intra: Max. radius of temporal edges
        Returns:
            cii: Temporal nodes i (from) of shape (N, 2) where N is the number of new edges
            cjj: Temporal nodes j (to) of shape (N, 2) where N is the number of new edges
        """
        cii, cjj = [], []
        ix = torch.arange(t - dt_intra, t, device=self.device)
        jx = torch.arange(t - dt_intra, t, device=self.device)
        ii, jj = torch.meshgrid(ix, jx)
        ii, jj = ii.reshape(-1), jj.reshape(-1)
        mask = (ii == jj) | (ii < 0) | (jj < 0) | ((ii - jj).abs() > r_intra)
        ii, jj = ii[~mask], jj[~mask]
        if not mask.all():
            for cam in range(self.frame_buffer.n_cams):
                cc = torch.ones((ii.shape[0]), dtype=torch.long, device=self.device) * cam
                cii.append(torch.stack([ii, cc], 1))  # [[t, c], ....
                cjj.append(torch.stack([jj, cc], 1))  # [[t, c], ....
        return torch.cat(cii, 0), torch.cat(cjj, 0)

    def get_spatial_temporal_edges(
            self,
            t: int,
            dt_inter: int,
            r_inter: int,
            dt_stereo: int,
            vel_vec: Optional[torch.Tensor] = None
    ) -> Tuple[CamNodeList, CamNodeList]:
        """ Returns spatial and spatial-temporal edges
        Args:
            t: Current time step
            dt_inter: Time-window in which spatial-temporal edges are added
            r_inter: Radius of spatial-temporal edges
            dt_stereo: Time-window in which spatial edges are added
            vel_vec: Current velocity vector in camera-reference frame of shape (3) - Default: [0., 0., 1.]
        Returns:
            cii: Spatial temporal nodes i (from) of shape (N, 2) where N is the number of new edges
            cjj: Spatial temporal nodes j (to) of shape (N, 2) where N is the number of new edges
        """
        cii, cjj = [], []
        vel_vec = vel_vec if vel_vec is not None else torch.tensor([0., 0., 1.], device=self.device)
        proj_vel = self.get_velocity_proj(vel_vec=vel_vec)
        for cam_i, cams_j in self.get_neighborhood().items():
            for cam_j in cams_j:
                # add spatial edges
                ii = torch.arange(t - dt_stereo, t, device=self.device)
                cci = torch.ones((ii.shape[0]), dtype=torch.long, device=self.device) * cam_i
                ccj = torch.ones((ii.shape[0]), dtype=torch.long, device=self.device) * cam_j
                cii.append(torch.stack([ii, cci], 1))
                cjj.append(torch.stack([ii, ccj], 1))   # For spatial edges: ii == jj

                # add spatial temporal edges
                if proj_vel[cam_i] <= proj_vel[cam_j] and dt_inter >= 1:
                    ii = torch.arange(t - dt_inter, t, device=self.device)
                    jj = ii - r_inter
                    mask = (ii >= max(t - dt_inter - 1, 0)) & (jj >= max(t - dt_inter - 1, 0))
                    if mask.any():
                        ii, jj = ii[mask], jj[mask]
                        cci = torch.ones((sum(mask)), dtype=torch.long, device=self.device) * cam_i
                        ccj = torch.ones((sum(mask)), dtype=torch.long, device=self.device) * cam_j
                        cii.append(torch.stack([ii, cci], 1))
                        cjj.append(torch.stack([jj, ccj], 1))
                        cii.append(torch.stack([jj, ccj], 1))
                        cjj.append(torch.stack([ii, cci], 1))
        return torch.cat(cii, 0), torch.cat(cjj, 0)

    def get_proximity_edges(
        self,
        t0: Optional[int] = 0,
        t1: Optional[int] = 0,
        rad: Optional[int] = 2,
        nms: Optional[int] = 2,
        thresh: Optional[float] = 16.0,
    ) -> Tuple[CamNodeList, CamNodeList]:
        """ Returns edges based on temporal proximity """
        cii, cjj = [], []
        for c in range(self.frame_buffer.n_cams):
            t = self.frame_buffer.counter
            ix = torch.arange(t0, t)
            jx = torch.arange(t1, t)

            ii, jj = torch.meshgrid(ix, jx)
            ii, jj = ii.reshape(-1), jj.reshape(-1)
            cci = torch.ones_like(ii) * c
            d = self.frame_buffer.distance(
                torch.stack([ii, cci], 1),
                torch.stack([jj, cci], 1),
            )
            d[ii - rad < jj] = np.inf
            d[d > 100] = np.inf

            ii1 = self.cii[self.cii[:, 1] == c, 0]
            jj1 = self.cjj[self.cjj[:, 1] == c, 0]
            for i, j in zip(ii1.cpu().numpy(), jj1.cpu().numpy()):
                for di in range(-nms, nms + 1):
                    for dj in range(-nms, nms + 1):
                        if abs(di) + abs(dj) <= max(min(abs(i - j) - 2, nms), 0):
                            i1 = i + di
                            j1 = j + dj

                            if (t0 <= i1 < t) and (t1 <= j1 < t):
                                d[(i1 - t0) * (t - t1) + (j1 - t1)] = np.inf

            es = []
            for i in range(t0, t):
                for j in range(max(i - rad - 1, 0), i):
                    es.append((i, j))
                    es.append((j, i))
                    d[(i - t0) * (t - t1) + (j - t1)] = np.inf

            ix = torch.argsort(d)
            for k in ix:
                if d[k].item() > thresh:
                    continue

                if len(es) > self.n_edges_max:
                    break

                i = ii[k]
                j = jj[k]

                # bidirectional
                es.append((i, j))
                es.append((j, i))

                for di in range(-nms, nms + 1):
                    for dj in range(-nms, nms + 1):
                        if abs(di) + abs(dj) <= max(min(abs(i - j) - 2, nms), 0):
                            i1 = i + di
                            j1 = j + dj

                            if (t0 <= i1 < t) and (t1 <= j1 < t):
                                d[(i1 - t0) * (t - t1) + (j1 - t1)] = np.inf

            ii, jj = torch.as_tensor(es, device=self.device).unbind(dim=-1)
            cci = torch.ones_like(ii) * c
            cii.append(torch.stack([ii, cci], 1))
            cjj.append(torch.stack([jj, cci], 1))
        return torch.cat(cii, 0), torch.cat(cjj, 0)

    def add_static_edges(
            self,
            t: int,
            dt_intra: int,
            dt_inter: int,
            r_intra: int,
            r_inter: int,
            dt_stereo: int,
            vel_vec: Optional[torch.Tensor] = None
    ) -> None:
        """ Add edges in a static manner
        Args:
            t: Current time step
            dt_intra: Time-window in which temporal edges are added
            dt_inter: Time-window in which spatial-temporal edges are added
            r_intra: Max. radius of temporal edges
            r_inter: Radius of spatial-temporal edges
            dt_stereo: Time-window in which spatial edges are added
            vel_vec: Current velocity vector in camera-reference frame of shape (3) - Default: [0., 0., 1.]
        """
        # add temporal edges
        cii, cjj = self.get_temporal_edges(t=t, dt_intra=dt_intra, r_intra=r_intra)

        # add spatial and spatial-temporal edges
        if self.frame_buffer.n_cams > 1:
            cii_spat_temp, cjj_spat_temp = self.get_spatial_temporal_edges(
                t=t,
                dt_inter=dt_inter,
                r_inter=r_inter,
                dt_stereo=dt_stereo,
                vel_vec=vel_vec
            )
            cii = torch.cat([cii, cii_spat_temp], 0)
            cjj = torch.cat([cjj, cjj_spat_temp], 0)

        # --- ToDo: Debugging ---
        # a = torch.stack([self.flatten_nodes(cii), self.flatten_nodes(cjj)])
        # a = (a[1].max() + 1) * a[0] + a[1]
        # idx = a.sort()[1]
        # cii, cjj = cii[idx], cjj[idx]
        # -----------------------

        # remove edges
        existing_edges = torch.zeros(len(self.cii), device=self.device, dtype=torch.bool)
        for i, (ci, cj) in enumerate(zip(self.cii, self.cjj)):
            existing_edges[i] = ((cii == ci).all(1) & (cjj == cj).all(1)).any()
        if len(existing_edges) > 0 and not existing_edges.all():
            self.rm_factors(~existing_edges)

        # add edges
        new_edges = torch.ones(len(cii), device=self.device, dtype=torch.bool)
        if len(self.cii) > 0:
            for i, (ci, cj) in enumerate(zip(cii, cjj)):
                new_edges[i] = ((self.cii != ci).any(1) | (self.cjj != cj).any(1)).all()
        self.add_factors(cii[new_edges], cjj[new_edges])

    def add_neighborhood_edges(
            self,
            t0: int,
            t1: int,
            r_intra: Optional[int] = 3,
            r_inter: Optional[int] = 2,
            vel_vec: Optional[torch.Tensor] = None,
    ) -> None:
        """ Add edges between neighboring frames within radius r """
        cii, cjj = self.get_temporal_edges(t=t1, dt_intra=t1-t0, r_intra=r_intra)
        if self.frame_buffer.n_cams > 1:
            cii_spat_temp, cjj_spat_temp = self.get_spatial_temporal_edges(
                t=t1,
                dt_inter=t1 - t0 - r_inter,
                r_inter=r_inter,
                dt_stereo=t1 - t0,
                vel_vec=vel_vec
            )
            cii = torch.cat([cii, cii_spat_temp], 0)
            cjj = torch.cat([cjj, cjj_spat_temp], 0)
        self.add_factors(cii, cjj)

    def add_proximity_edges(
            self,
            t0: Optional[int] = 0,
            t1: Optional[int] = 0,
            r_intra: Optional[int] = 2,
            r_inter: Optional[int] = 2,
            nms: Optional[int] = 2,
            thresh: Optional[float] = 16.0,
            remove: Optional[bool] = False,
            vel_vec: Optional[torch.Tensor] = None,
    ) -> None:
        """ Add edges to the factor graph based on distance """
        cii, cjj = self.get_proximity_edges(
            t0=t0,
            t1=t1,
            rad=r_intra,
            nms=nms,
            thresh=thresh,
        )
        if self.frame_buffer.n_cams > 1:
            t = self.frame_buffer.counter
            cii_spat_temp, cjj_spat_temp = self.get_spatial_temporal_edges(
                t=t,
                dt_inter=t - t0 - r_inter,
                r_inter=r_inter,
                dt_stereo=t - t0,
                vel_vec=vel_vec
            )
            cii = torch.cat([cii, cii_spat_temp], 0)
            cjj = torch.cat([cjj, cjj_spat_temp], 0)
        self.add_factors(cii, cjj, remove)

    def get_confidence_maps(self, idx: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        """ Computes depth confidence from flow confidence
        Results:
            confidence_maps: Depth confidence from edges-- (N, C, C, H, W)
            stats: Confidence statistics used to determine if geom. prior established metric scale
        """
        intra_inter_conf_maps = []
        for ci in range(self.frame_buffer.n_cams):
            mask = (self.cii[:, 0] == idx) & (self.cii[:, 1] == ci)
            mask_intra = mask & (self.cii[:, 1] == self.cjj[:, 1])
            mask_inter = mask & (self.cii[:, 1] != self.cjj[:, 1])
            mean_weight = self.weight.mean(-1)
            intra_inter_conf_maps.append(torch.cat([
                mean_weight[0, mask_intra].max(dim=0, keepdim=True)[0],
                mean_weight[0, mask_inter].max(dim=0, keepdim=True)[0],
            ]))
        intra_inter_conf_maps = torch.stack(intra_inter_conf_maps)
        conf_maps = intra_inter_conf_maps.max(dim=1, keepdim=False)[0]
        stats = {
            'total_conf_ratio': float(conf_maps.mean()),
            'intra_conf_ratio': float(intra_inter_conf_maps[:, 0].mean()),
            'inter_conf_ratio': float(intra_inter_conf_maps[:, 1].mean()),
            'intra_x_inter_conf_ratio': float((intra_inter_conf_maps[:, 0] * intra_inter_conf_maps[:, 1]).mean()),
        }
        return conf_maps, stats
