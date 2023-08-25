from __future__ import annotations

import torch
import numpy as np

from typing import Optional, List, Dict, Tuple, Union
from typing_extensions import TypedDict, NotRequired
import numpy.typing as npt
from PIL.Image import Image


class Boxes3D(TypedDict):
    """
    idx: Tracking ID of N objects. Consistent with instance map -- (N, 1)
    dim: (width, height, length) for N objects -- (N, 3)
    pos: (x, y, z), pos. of N objects in world coordinates (center of bottom face of 3D bounding box) -- (N, 3)
    rot: (rx, ry, rz), of N obj. in world coord. ry == 0 iff obj. is aligned with x-ax. and pointing right -- (N, 3)
    """
    idx: npt.NDArray[np.int]
    dim: npt.NDArray[np.float]
    pos: npt.NDArray[np.float]
    rot: npt.NDArray[np.float]


class MetaData(TypedDict):
    """Input metadata.
    url: Relative path from sample to dataset location
    video_name: Name of video / scene
    index: Index of item in video / scene
    timestamp: Time at which sample was observed
    """
    url: NotRequired[str | None]
    video_name: str
    frame_index: int
    timestamp: NotRequired[float | None]


class Frame(TypedDict):
    """
    metadata: Frame Metadata
    rgb: Relative path from dataset to rgb image
    intrinsics: (fx, fx, cx, cy)
    pose: cam2world (4, 4)
    mask: Path to mask. Can be used to mask non-informative parts of rgb / depth, e.g. self occlusion
    depth: Relative path from dataset to depth map
    fwd_opt_flow: Relative path from dataset to forward opt. flow map
    bwd_opt_flow: Relative path from dataset to backward opt. flow map
    fwd_scene_flow: Relative path from dataset to forward scene flow map
    bwd_scene_flow: Relative path from dataset to backward scene flow map
    instance_map: path to instance map
    others: Dict of other, not typed data
    """
    metadata: MetaData
    rgb: str
    intrinsics: npt.NDArray[np.float]
    pose: NotRequired[npt.NDArray[np.float] | None]
    depth: NotRequired[str | None]
    mask: NotRequired[npt.NDArray[Union[str, np.uint8]] | None]
    fwd_opt_flow: NotRequired[str | None]
    bwd_opt_flow: NotRequired[str | None]
    fwd_opt_flow_mask: NotRequired[str | None]
    bwd_opt_flow_mask: NotRequired[str | None]
    fwd_scene_flow: NotRequired[str | None]
    bwd_scene_flow: NotRequired[str | None]
    instance_map: NotRequired[npt.NDArray[np.int] | None]
    semantic_map: NotRequired[npt.NDArray[np.int] | None]
    others: NotRequired[Dict[str, npt.NDArray] | None]


class Group(TypedDict):
    """
    ego_pose: ego2world 4x4 transformation matrix
    lidar: path to lidar data
    t_lidar: Lidar timestamp
    lidar_pose: lidar2ego 4x4 transformation matrix
    obj_pos: List[Obj3D]
    frames: Dict[CamData]
    """
    metadata: MetaData
    group_pose: NotRequired[torch.Tensor | None]
    lidar: NotRequired[str | None]
    lidar_pose: NotRequired[torch.Tensor | None]
    boxes3d: NotRequired[Boxes3D | None]
    frames: Dict[Union[str, int], Frame]


Dataset = Dict[str, List[Group]]


class Sample(TypedDict):
    """
    sequence: Sequence name
    tgt: Target group / frame
    ref: Reference group / frame (temporal context)
    """
    sequence: str
    groups: Dict[int, Group]


class InputData(TypedDict):
    """Container holding the input data.
    idx: index of sample -- (1)
    tag: dataset tag ToDo: Remove?
    filename: Filename of each file in temp. context {-1: [$path], 0: [$path], ...}
    splitname: ToDo: Remove?
    sensor_name: ['Camera_0', 'Camera_1']  # ToDo standardize
    rgb: RGB images N: Num ref. views -- {context: (N, 3, H, W)}
    intrinsics: ...
    extrinsics: ...
    depth: Gt depth -- {context: (N, 1, H, W)}
    instance_segmentation: ...
    boxes2d: ...
    boxes3d: ...
    filename_context: ...
    """
    idx: torch.Tensor
    scene: str
    filename: Dict[int, List[str]]
    rgb: Dict[int, torch.Tensor]
    intrinsics: Dict[int, torch.Tensor]
    extrinsics: Dict[int, torch.Tensor]
    depth: NotRequired[Dict[int, torch.Tensor] | None]
    instance_segmentation: NotRequired[Dict[int, torch.Tensor] | None]
    bboxes3d: NotRequired[Dict[int, Dict[str, npt.NDArray[np.float32]]] | None]
    points: NotRequired[Dict[int, npt.NDArray[np.float32]] | None]


class InputFrame(TypedDict):
    """
    idx: Index of sample in whole dataset, i.e. idx in {0, ..., len(dataset)}
    frame_idx: Index of sample in scene, i.e. frame_idx in {0, ..., len(scene)}
    cam: Camera name
    filename:
    rgb:
    intrinsics: Intrinsics array -- {ctx: (3,3)}
    pose: Frame pose (frame2world) -- {ctx: (4, 4)}
    depth: Sparse or dense pixle-wise depth (0 = not available) -- {ctx: (H, W)}
    mask: Mask for hiding non-informative areas (e.g. self-occlusion) (False: not-informative) -- {ctx: (H, W)}
    instance: Instance map: Pixel value = inst. ID, pixel value = 0 -> background -- {ctx: (H, W)}
    fwd_optical_flow: Opt. flow from frame t to t-1 -- {ctx: (H, W, 2)}
    bwd_optical_flow: Opt. flow from frame t to t+1 -- {ctx: (H, W, 2)}
    bwd_flow_mask:
    fwd_flow_mask:
    bwd_scene_flow: Scene flow from frame t to t-1 -- {ctx: (H, W, 3)}
    fwd_scene_flow: Scene flow from frame t to t+1 -- {ctx: (H, W, 3)}
    semantic: Semantic segmentation map: Pixel value = sem. ID, p
    """
    idx: int
    frame_idx: Dict[int, int]
    cam: str
    filename: Dict[int, str]
    rgb: Dict[int, Image]
    intrinsics: Dict[int, npt.NDArray[np.float]]
    pose: NotRequired[Dict[int, npt.NDArray[np.float]] | None]
    depth: NotRequired[Dict[int, npt.NDArray[np.float]] | None]
    mask: NotRequired[Dict[int, npt.NDArray[np.float]] | None]
    instance: NotRequired[Dict[int, npt.NDArray[np.int]] | None]
    fwd_optical_flow: NotRequired[Dict[int, npt.NDArray[np.float]] | None]
    bwd_optical_flow: NotRequired[Dict[int, npt.NDArray[np.float]] | None]
    bwd_flow_mask: NotRequired[Dict[int, np.ndarray] | None]
    fwd_flow_mask: NotRequired[Dict[int, np.ndarray] | None]
    bwd_scene_flow: NotRequired[Dict[int, npt.NDArray[np.float]] | None]
    fwd_scene_flow: NotRequired[Dict[int, npt.NDArray[np.float]] | None]
    semantic: NotRequired[Dict[int, npt.NDArray[np.int]] | None]
    others: NotRequired[Dict[int, Dict[str, npt.NDArray[np.int]]] | None]
