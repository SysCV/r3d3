from __future__ import annotations
from typing_extensions import TypedDict, NotRequired
from typing import Optional, Dict, Tuple

import torch

from r3d3.r3d3_net import R3D3Net
from r3d3.frame_buffer import FrameBuffer, CompletionMode
from r3d3.startup_filter import StartupFilter, ImageMode
from r3d3.process import R3D3Process, GraphType
from r3d3.covis_graph import CorrelationMode
from r3d3.modules.completion import DepthCompletion

from collections import OrderedDict


class R3D3Output(TypedDict):
    """Input metadata.
    disp: Estimated low resolution disparity of shape (C, H/8, W/8)
    disp_up: Estimated full resolution disparity of shape (C, H, W)
    conf: Estimated low resolution confidence of shape (C, H/8, W/8)
    pose: Estimated pose of shape (7)
    """
    disp: NotRequired[torch.Tensor | None]
    disp_up: NotRequired[torch.Tensor | None]
    conf: NotRequired[torch.Tensor | None]
    pose: NotRequired[torch.Tensor | None]
    stats: NotRequired[Dict[str, float] | None]


class R3D3:
    def __init__(
            self,
            n_cams: int,
            weights: str,
            image_size: Tuple[int, int],
            scale: Optional[float] = 1.0,
            completion_net: Optional[DepthCompletion] = None,
            completion_mode: Optional[CompletionMode] = CompletionMode.ALL,
            buffer_size: Optional[int] = 0,
            filter_thresh: Optional[float] = 2.5,
            depth_init: Optional[float] = 1.0,
            image_mode: Optional[ImageMode] = ImageMode.RGB,
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
            n_cams: Number of cameras
            weights: Path to encoder & GRU weights
            image_size: Image dimensions (Height, Width)
            scale: Scale at which R3D3 operates (metric-scale / scale)
            completion_net: Completion network instance. Defaut: None - No completion is used
            completion_mode: Which frames should be completed
            buffer_size: Size of R3D3 frame buffer (=# timesteps). Default: 0 - buffer size depending on ref-window
            filter_thresh: Warmup frames are kept if mean-flow(current -> prev. frame) > filter_thresh
            depth_init: Value with which depth is initialized
            image_mode: Image-mode to be used for frame-encoders
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
        super(R3D3, self).__init__()
        self.completion_net = completion_net
        if completion_net is None:
            completion_mode = CompletionMode.NONE
        self.r3d3_net = R3D3Net()
        self.load_weights(weights)

        self.scale = scale

        # store images, depth, poses, intrinsics (shared between processes)
        self.frame_buffer = FrameBuffer(
            image_size,
            buffer_size if buffer_size > 0 else max(ref_window + 1, n_warmup + 2),
            n_cams=n_cams,
            scale=self.scale,
            completion_net=self.completion_net,
            completion_mode=completion_mode,
        )
        # filter incoming frames so that there is enough motion
        self.startup_filter = StartupFilter(
            r3d3_net=self.r3d3_net,
            frame_buffer=self.frame_buffer,
            startup_thresh=filter_thresh,
            depth_init=depth_init / self.scale,
            image_mode=image_mode,
        )

        self.process = R3D3Process(
            net=self.r3d3_net,
            frame_buffer=self.frame_buffer,
            init_motion_only=init_motion_only,
            iters_init=iters_init,
            iters1=iters1,
            iters2=iters2,
            n_warmup=n_warmup,
            frame_thresh=frame_thresh,
            optm_window=optm_window,
            disable_comp_inter_flow=disable_comp_inter_flow,
            corr_impl=corr_impl,
            n_edges_max=n_edges_max,
            graph_type=graph_type,
            ref_window=ref_window,
            proximity_thresh=proximity_thresh,
            nms=nms,
            max_age=max_age,
            dt_intra=dt_intra,
            dt_inter=dt_inter,
            r_intra=r_intra,
            r_inter=r_inter,
        )

    def load_weights(self, weights):
        """ load trained model weights """
        loaded_weights = torch.load(weights)
        state_dict = loaded_weights['state_dict'] if 'state_dict' in loaded_weights else loaded_weights
        state_dict = OrderedDict([
            (k.replace("module.", "").replace("networks.droid_net.", ""), v) for (k, v) in state_dict.items()])

        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

        self.r3d3_net.load_state_dict(state_dict, strict=False)
        self.r3d3_net.to("cuda:0").eval()

    def track(
            self,
            tstamp,
            image,
            mask=None,
            intrinsics=None,
            pose_rel=None,
            pose=None
    ) -> R3D3Output:
        """ main thread - update map """
        with torch.no_grad():
            if pose_rel is not None:
                pose_rel[..., :3] /= self.scale
            if pose is not None:
                pose[..., :3] /= self.scale
            self.startup_filter.track(
                timestamp=tstamp,
                image=image,
                intrinsics=intrinsics,
                mask=mask,
                pose_rel=pose_rel,
                pose=pose,
                initialize=not self.process.is_initialized
            )

            # monocular predictions for initial frames
            disp_up, disp = None, None
            if self.frame_buffer.counter <= self.process.n_warmup + 1 and \
                    self.frame_buffer.completion_mode is not CompletionMode.NONE:
                if self.startup_filter.skip_count == 0:    # frame was added to buffer, use buffer data
                    self.frame_buffer.mono_from_buffer(t=self.frame_buffer.counter)
                    disp_up = self.frame_buffer.disps_up[self.frame_buffer.counter - 1] / self.scale
                else:   # frame was not added to buffer, use input instead
                    disp_up, disp = self.frame_buffer.mono_from_input(
                        image=image / 255.0,
                        intrinsics=intrinsics[:, 0]
                    )

            self.process()  # R3D3 inference

            # gather output
            output: R3D3Output = {
                'disp': disp,
                'disp_up': disp_up,
                'conf': None,
                'pose': None,
                'stats': None,
            }
            if self.frame_buffer.counter > self.process.n_warmup:  # Check if in active inference phase
                idx = self.frame_buffer.counter - 1
                conf_maps, conf_stats = self.process.graph.get_confidence_maps(idx)
                output['disp'] = self.frame_buffer.disps[idx] / self.scale
                output['disp_up'] = self.frame_buffer.disps_up[idx] / self.scale
                output['conf'] = conf_maps
                output['pose'] = self.frame_buffer.poses[idx].clone()
                output['pose'][:3] *= self.scale
                output['stats'] = conf_stats
        return output

