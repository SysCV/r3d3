from typing import Optional
from enum import Enum

import torch

from r3d3.r3d3_net import R3D3Net
from r3d3.frame_buffer import FrameBuffer
from r3d3.covis_graph import CovisGraph, CorrelationMode
from r3d3.modules.corr import AltCorrBlock


class GraphType(Enum):
    STATIC = 'static'
    DROID_SLAM = 'droid_slam'


class R3D3Process:
    def __init__(
            self,
            net: R3D3Net,
            frame_buffer: FrameBuffer,
            init_motion_only: Optional[bool] = True,
            iters_init: Optional[bool] = 8,
            iters1: Optional[bool] = 4,
            iters2: Optional[bool] = 2,
            n_warmup: Optional[int] = 3,
            frame_thresh: Optional[float] = 2.25,
            optm_window: Optional[int] = 1,
            disable_comp_inter_flow: Optional[bool] = False,
            corr_impl: Optional[CorrelationMode] = CorrelationMode.VOLUME,
            n_edges_max: Optional[int] = -1,
            graph_type: Optional[GraphType] = GraphType.STATIC,
            ref_window: Optional[int] = 25,
            proximity_thresh: Optional[int] = 12.0,
            nms: Optional[int] = 1,
            max_age: Optional[int] = 25,
            dt_intra: Optional[int] = 3,
            dt_inter: Optional[int] = 2,
            r_intra: Optional[int] = 2,
            r_inter: Optional[int] = 2,
    ):
        """
        Args:
            net: GRU module instance
            frame_buffer: Frame buffer instance
            init_motion_only: If depth should be fixed during the first "iters_init"-DBA iterations
            iters_init: Number of initialization DBA+GRU iterations - (2x"iters_init" total iterations)
            iters1: Number of DBA+GRU iterations after adding a new frame
            iters2: Number of additional DBA+GRU iterations if previous timestep was removed
            n_warmup: Number of warmup timesteps
            frame_thresh: Prev. frame (at t-1) is removed if mean-induced-flow(t-1 -> t-2) of front-cam is < thresh
            optm_window: Time-window in which pose & depth is optimized (for older timesteps only depth is optimized)
            disable_comp_inter_flow: If rotational flow compensation should be disabled
            corr_impl: Type of correlation volume implementation
            n_edges_max: Max. allowed number of edges (remove old edges if too many)
            graph_type: What covisibility graph construction algorithm to use
            ref_window: Time window in which depth is optimized
            proximity_thresh: Threshold for proximity based edges in Droid-SLAM graph implementation
            nms: "Non-maximum-supperession" based limit of distance between nodes in Droid-SLAM graph implementation
            max_age: Maximum edge age in Droid-SLAM graph implementation
            dt_intra: Time-window in which temporal edges are added
            dt_inter: Time-window in which spatial-temporal edges are added
            r_intra: Max. radius of temporal edges
            r_inter: Radius of spatial temporal edges
        """
        self.frame_buffer = frame_buffer
        self.update_op = net.update

        self.t0 = 0
        self.t1 = 0

        self.is_initialized = False
        self.count = 0

        self.init_motion_only = init_motion_only
        self.iters_init = iters_init
        self.iters1 = iters1
        self.iters2 = iters2
        self.n_edges_max = n_edges_max

        self.frame_thresh = frame_thresh
        self.n_warmup = n_warmup

        self.optm_window = optm_window
        self.ref_window = ref_window if graph_type == GraphType.DROID_SLAM else max(dt_intra, dt_inter + 1) # ToDo: Check

        self.graph_type = graph_type

        # Droid-SLAM graph settings
        self.nms = nms
        self.max_age = max_age
        self.proximity_thresh = proximity_thresh

        # Static graph settings
        self.dt_intra = dt_intra
        self.dt_inter = dt_inter
        self.r_intra = r_intra
        self.r_inter = r_inter

        self.graph = CovisGraph(
            frame_buffer=frame_buffer,
            update_op=net.update,
            n_edges_max=n_edges_max,
            corr_impl=corr_impl,
            disable_comp_inter_flow=disable_comp_inter_flow,
        )

    def __update(self) -> None:
        """ Add edges -> perform GRU & BA updates -> remove prev. frame if too close -> depth completion """
        self.count += 1
        self.t1 += 1
        upmask, weight, cii = None, None, None

        if self.graph.corr_impl == CorrelationMode.LOWMEM:
            self.graph.corr = AltCorrBlock(self.frame_buffer.corr_feat_flat[None])

        if self.graph_type == GraphType.DROID_SLAM:
            if self.graph.corr is not None:
                self.graph.rm_factors(self.graph.age > self.max_age)
            self.graph.add_proximity_edges(
                t0=max(self.t1 - 5, 0),
                t1=max(self.t1 - self.ref_window, 0),
                r_intra=self.r_intra,
                r_inter=self.r_inter,
                nms=self.nms,
                thresh=self.proximity_thresh,
                remove=True,
                vel_vec=None
            )
        elif self.graph_type == GraphType.STATIC:
            self.graph.add_static_edges(
                t=self.t1,
                dt_intra=self.dt_intra,
                dt_inter=self.dt_inter,
                r_intra=self.r_intra,
                r_inter=self.r_inter,
                dt_stereo=self.optm_window,
                vel_vec=None
            )

        for itr in range(self.iters1):
            weight, upmask, cii = self.graph.update(
                t0=max(1, self.t1 - self.optm_window),
                t1=self.t1
            )

        if self.frame_buffer.counter > self.n_warmup + 1:    # Complete if geom. depth useful
            self.frame_buffer.upsample(torch.unique(cii, dim=0), upmask)
            self.frame_buffer.complete(
                t=self.t1,
                weight=weight,
                cii=cii,
                t_from_ref=self.t1 - self.ref_window,
                t_from_opt=self.t1 - self.optm_window
            )

        # Remove previous timestep if too close
        d = self.frame_buffer.distance(torch.tensor([[self.t1 - 3, 0]]), torch.tensor([[self.t1 - 2, 0]]))
        if d.item() < self.frame_thresh:
            self.graph.rm_keyframe(self.t1 - 2)
            self.frame_buffer.counter -= 1
            self.t1 -= 1

            # restore graph without the removed timestep
            if self.graph_type == GraphType.STATIC:
                self.graph.add_static_edges(
                    t=self.t1,
                    dt_intra=self.dt_intra,
                    dt_inter=self.dt_inter,
                    r_intra=self.r_intra,
                    r_inter=self.r_inter,
                    dt_stereo=self.optm_window,
                    vel_vec=None
                )
        else:
            for itr in range(self.iters2):
                weight, upmask, cii = self.graph.update(
                    t0=max(1, self.t1 - self.optm_window),
                    t1=self.t1
                )

            self.frame_buffer.upsample(torch.unique(cii, dim=0), upmask)
            self.frame_buffer.complete(
                t=self.t1,
                weight=weight,
                cii=cii,
                t_from_ref=self.t1 - self.ref_window,
                t_from_opt=self.t1 - self.optm_window
            )

        if self.t1 >= self.frame_buffer.buffer:    # Clear oldest frame if frame buffer is full
            self.graph.rm_first_frame()
            self.frame_buffer.counter -= 1
            self.t1 -= 1

        # initialize pose for next timestep
        self.frame_buffer.poses[self.t1] = self.frame_buffer.poses[self.t1 - 1].clone()
        prev_disps = self.frame_buffer.disps[max(0, self.t1 - 4):self.t1]
        self.frame_buffer.disps[self.t1] = prev_disps.mean(-1, keepdim=True).mean(-2, keepdim=True).mean(0)

        # update visualization
        self.frame_buffer.dirty[self.graph.cii[:, 0].min():self.t1] = True

    def __initialize(self) -> None:
        """ Initialize the system """

        self.t0 = 0
        self.t1 = self.frame_buffer.counter

        if self.graph.corr_impl == CorrelationMode.LOWMEM:
            self.graph.corr = AltCorrBlock(self.frame_buffer.corr_feat_flat[None])

        if self.graph_type == GraphType.DROID_SLAM:
            self.graph.add_neighborhood_edges(
                t0=self.t0,
                t1=self.t1,
                r_intra=3,
                r_inter=self.r_inter,
                vel_vec=None,
            )
        elif self.graph_type == GraphType.STATIC:
            self.graph.add_static_edges(
                t=self.t1,
                dt_intra=self.dt_intra,
                dt_inter=self.dt_inter,
                r_intra=self.r_intra,
                r_inter=self.r_inter,
                dt_stereo=self.n_warmup,
                vel_vec=None
            )

        for itr in range(self.iters_init):
            weight, _, cii = self.graph.update(t0=1, t1=self.t1, motion_only=self.init_motion_only)

        if self.graph_type == GraphType.DROID_SLAM:
            self.graph.add_proximity_edges(
                t0=0,
                t1=0,
                r_intra=2,
                r_inter=self.r_inter,
                nms=2,
                thresh=self.proximity_thresh,
                remove=True,
                vel_vec=None
            )

        for itr in range(self.iters_init):
            weight, _, cii = self.graph.update(t0=1, t1=self.t1)

        # initialize pose for next timestep
        self.frame_buffer.poses[self.t1] = self.frame_buffer.poses[self.t1 - 1].clone()
        prev_disps = self.frame_buffer.disps[self.t1 - self.n_warmup:self.t1]
        self.frame_buffer.disps[self.t1] = prev_disps.mean(-1, keepdim=True).mean(-2, keepdim=True).mean(0)

        self.frame_buffer.dirty[:self.t1] = True

        if self.graph_type == GraphType.DROID_SLAM:
            self.graph.rm_factors(self.graph.cii[:, 0] < self.n_warmup - 4)
        self.is_initialized = True

    def __call__(self) -> None:
        """ Main update method -> call at each timestep! """

        if not self.is_initialized and self.frame_buffer.counter == self.n_warmup:
            self.__initialize()
        elif self.is_initialized and self.t1 < self.frame_buffer.counter:
            self.__update()
