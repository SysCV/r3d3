from typing import Optional, Tuple, Union
from enum import Enum

import torch
from lietorch import SE3

from r3d3.r3d3_net import R3D3Net
from r3d3.frame_buffer import FrameBuffer
import r3d3.geom.projective_ops as pops
from r3d3.modules.corr import CorrBlock


class ImageMode(Enum):
    RGB = 'rgb'
    BGR = 'bgr'


class StartupFilter:
    """ This class is used to filter incoming frames and extract features """
    def __init__(
            self,
            r3d3_net: R3D3Net,
            frame_buffer: FrameBuffer,
            startup_thresh: Optional[float] = 2.5,
            device: Union[torch.device, str, None] = "cuda:0",
            depth_init: Optional[float] = 1.0,
            image_mode: Optional[ImageMode] = ImageMode.RGB
    ):

        """
        Args:
            r3d3_net: R3D3-Net instance with feature and context encoders
            frame_buffer: Frame-Buffer instance where data is stored
            startup_thresh: Threshold on mean flow which needs to be surpassed to add frame
            device: Torch device on which to run
            depth_init: Initial depth value
            image_mode: If images should be fed as RGB or BGR into encoder
        """
        self.context_net = r3d3_net.cnet
        self.feature_net = r3d3_net.fnet
        self.update = r3d3_net.update

        self.frame_buffer = frame_buffer
        self.thresh = startup_thresh
        self.device = device

        self.skip_count = 0

        self.disp_init = 1 / depth_init
        self.image_mode = image_mode

        # mean, std for image normalization
        self.mean = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        self.std = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]

        self.hidden_init, self.ctx_feat, self.corr_feat = None, None, None

    @torch.cuda.amp.autocast(enabled=True)
    def __context_encoder(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Features for GRU
        Args:
            image: Normalized input image of shape (ToDo)
        Returns:
            hidden_init: GRU hidden state initialization features
            ctx_feat: Context features
        """
        """ context features """
        hidden_init, ctx_feat = self.context_net(image).split([128, 128], dim=2)
        return hidden_init.tanh(), ctx_feat.relu()

    @torch.cuda.amp.autocast(enabled=True)
    def __feature_encoder(self, image: torch.Tensor) -> torch.Tensor:
        """ Features for correlation volume
        Args:
            image: Normalized input image of shape (ToDo)
        Returns:
            correlation volume features
        """
        return self.feature_net(image)

    def get_skip_count(self) -> int:
        """ Returns # frames which have been skipped
        """
        return self.skip_count

    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def track(
            self,
            timestamp: float,
            image: torch.Tensor,
            intrinsics: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            pose_rel: Optional[torch.Tensor] = None,
            pose: Optional[torch.Tensor] = None,
            initialize: Optional[bool] = False
    ) -> None:
        """ Adds data to frame buffer - skips data if not yet initialized and mean flow to prev. frame < thresh
        Args:
            timestamp:
            image:
            intrinsics:
            mask:
            pose_rel:
            pose:
            initialize: If R3D3 is already initialized
        """
        """ main update operation - run on every frame in video """
        identity = SE3.Identity(1, ).data[0]
        height = image.shape[-2] // 8
        width = image.shape[-1] // 8

        # normalize images
        if self.image_mode == ImageMode.RGB:
            inputs = image[:, None] / 255.0
        elif self.image_mode == ImageMode.BGR:
            inputs = image[:, None, [2, 1, 0]] / 255.0
        inputs = inputs.sub_(self.mean).div_(self.std)

        # extract features
        corr_feat = self.__feature_encoder(inputs)

        # always add first frame to the depth video
        if self.frame_buffer.counter == 0:
            pose = pose if pose is not None else identity
            hidden_init, ctx_feat = self.__context_encoder(inputs[:, [0]])
            self.hidden_init, self.ctx_feat, self.corr_feat = hidden_init, ctx_feat, corr_feat  # store for next iter
            self.frame_buffer.append(
                timestamp=timestamp,
                image=image,
                pose_rel=pose_rel,
                intrinsics=intrinsics / 8.0,
                corr_feat=corr_feat,
                hidden_init=hidden_init[:, 0],
                ctx_feat=ctx_feat[:, 0],
                pose_init=pose,
                disp_init=self.disp_init,
                mask=mask
            )

        # only add new frame if there is enough motion (mean flow) and not yet initialized
        else:
            add_frame = True
            if initialize:
                disp_init = self.disp_init
                coords0 = pops.coords_grid(height, width, device=self.device)[None, None]
                corr = CorrBlock(self.corr_feat[0:1, [0]], corr_feat[0:1, [0]])(coords0)

                if pose is not None:    # If pose is given: Only add frame if sufficient motion
                    cnt = self.frame_buffer.counter
                    rel_motion = ((SE3(self.frame_buffer.poses[cnt - 1]) * SE3(pose).inv()).data[0:3] ** 2).sum().sqrt()
                    add_frame = rel_motion > disp_init * 4e-2
                else:
                    # One gru update = (rough) flow prediction
                    _, delta, weight = self.update(self.hidden_init[0:1], self.ctx_feat[0:1], corr)
                    add_frame = delta.norm(dim=-1).mean().item() > self.thresh
            else:
                disp_init = None

            if add_frame:
                self.skip_count = 0
                hidden_init, ctx_feat = self.__context_encoder(inputs[:, [0]])
                self.hidden_init, self.ctx_feat, self.corr_feat = hidden_init, ctx_feat, corr_feat
                self.frame_buffer.append(
                    timestamp=timestamp,
                    image=image,
                    pose_rel=pose_rel,
                    intrinsics=intrinsics / 8.0,
                    corr_feat=corr_feat,
                    hidden_init=hidden_init[:, 0],
                    ctx_feat=ctx_feat[:, 0],
                    pose_init=pose,
                    disp_init=disp_init,
                    mask=mask
                )
            else:
                self.skip_count += 1
