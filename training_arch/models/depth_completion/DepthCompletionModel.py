from typing import Dict, List, Tuple, Optional, Union

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import hflip

from vidar.arch.models.BaseModel import BaseModel
from vidar.utils.config import Config, cfg_has
from training_arch.utils.utils import BackprojectDepth, Project3D, get_smooth_loss, SSIM, intr_3x3_to_4x4
from r3d3.utils import pose_quaternion_to_matrix


# Monodepth2 format which is either:
#   {('xyz', cam, context, scale): [data]}
#   {('xyz', cam, context): [data]}
#   {('xyz', scale): [data]}
#  where:
#   'xyz': data-field name
#   cam: Camera -> 0 = camera0, 1 = camera1 etc.
#   context: Context w.r.t. sample t -> 0 = t, -1 = t-1 etc.
#   scale: Scale w.r.t. original scale -> (H / 2**scale, W / 2**scale)
Monodepth2Format = Dict[Union[Tuple[str, int, int, int], Tuple[str, int, int], Tuple[str, int]], torch.Tensor]
Losses = Dict[str, Union[float, torch.Tensor]]

# Output format of the form {ctx: [(b, c, 1, H, W), (b, c, 1, H/2, W/2), (b, c, 1, H/4, W/4), ...]}
# where: b - batchsize, c - number of cameras, H - height, W - width
DepthOutput = Dict[int, List[torch.Tensor]]


class DepthCompletionModel(BaseModel):
    """ Model for R3D3 completion network training.
    The model is based on Monodepth2 https://github.com/nianticlabs/monodepth2
    """
    def __init__(self, cfg: Config, **kwargs):
        """
        Args:
            cfg: Model configuration
                use_gt_pose: If ground-truth poses should be used (otherwise poses used from others->poses)
                contexts: Context to use for photometric loss: {cam: [context]}, default: [[-1, +1]]
                          e.g. [[-1, +1], [0], [0]] -> (cam0, t-1), (cam0, t+1), (cam1, t), (cam2, t)
                          where (cam0, t) is the target frame, cam-n is the n-th cam in the camera list
                col_jit_params: Random color jitter parameters: [brightness, contrast, saturation, hue]
                                default: [0.2, 0.2, 0.2, 0.05] ([0, 0, 0, 0] -> disabled)
                p_flip: Probability of flipping input to network horizontally, default: 0.5 (0 -> disabled)
                p_zero_inp: Probability of setting input depth & confidence to 0, default: 0.2 (0 -> disabled)
                disable_automasking: True: Disables automasking, Default: False
                confidence_thresh: Masks photometric loss where conf. < thresh, default: 0.0 (0 -> disabled)
                flow_cons_threshold: Masks photometric loss where flow-consistency < thresh, default: 3 (0 -> disabled)
                max_depth_threshold: Masks photometric loss where geom-depth > thresh, default: 200 (0 -> disabled)
                weight_tgt_cam: Weighting of photometric loss from target cameras, default = 1.0
                weight_ref_cam: Weighting of photometric loss from reference cameras, default = 0.1
                disparity_smoothness: Weighting of disparity smootness loss, default = 0.001
                weight_ssim: Weighting of SSIM loss, default = 0.85
                weight_l1: Weighting of L1 loss, default = 0.15
                backproject_dim: Image dimension [B, H, W] with which backprojection module should be initialized
        """
        super(DepthCompletionModel, self).__init__(cfg=cfg, **kwargs)
        self.use_gt_pose = cfg_has(cfg.model, 'use_gt_pose', False)
        contexts: List[List[int]] = cfg_has(cfg.model, 'contexts', [[-1, 1]])  # Define spatio-temporal context frames
        self.contexts: Dict[int, List[int]] = {cam: ctx for cam, ctx in enumerate(contexts)}

        self.num_scales = 4
        self.scales = [0, 1, 2, 3]

        # augmentation
        params = cfg_has(cfg.model, 'col_jit_params', [0.2, 0.2, 0.2, 0.05])
        self.col_jit = transforms.ColorJitter(
            brightness=params[0],
            contrast=params[1],
            saturation=params[2],
            hue=params[3]
        )
        self.p_flip = cfg_has(cfg.model, 'p_flip', 0.5)
        self.p_zero_inp = cfg_has(cfg.model, 'p_zero_inp', 0.2)

        # masking
        self.disable_automasking = cfg_has(cfg.model, 'disable_automasking', False)
        self.confidence_thresh = cfg_has(cfg.model, 'confidence_thresh', 0.0)
        self.flow_cons_threshold = cfg_has(cfg.model, 'flow_cons_threshold', 3)
        self.max_depth_threshold = cfg_has(cfg.model, 'max_depth_threshold', 0.0)

        # loss weights: Total Loss =
        self.weight_tgt_cam = cfg_has(cfg.model, 'weight_tgt_cam', 1.0)
        self.weight_ref_cam = cfg_has(cfg.model, 'weight_ref_cam', 0.1)
        self.disparity_smoothness = cfg_has(cfg.model, 'disparity_smoothness', 0.001)
        self.weight_ssim = cfg_has(cfg.model, 'weight_ssim', 0.85)
        self.weight_l1 = cfg_has(cfg.model, 'weight_l1', 0.15)

        b, h, w = cfg_has(cfg.model, 'backproject_dim', [4, 384, 640])
        self.backproject_depth = BackprojectDepth(b, h, w)
        self.project_3d = Project3D(b, h, w)
        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((h // s, w // s), interpolation=transforms.InterpolationMode.BILINEAR)
        self.ssim = SSIM()

    def get_max_depth_masks(self, inputs: Monodepth2Format, batch: Dict) -> Monodepth2Format:
        mask = batch['others'][0]['disp_up'][0] > 1 / self.max_depth_threshold
        if ('mask', 0, 0, 0) in inputs:
            mask &= inputs[('mask', 0, 0, 0)]
        return {('mask', 0, 0, 0): mask}

    def get_flow_consistency_masks(self, inputs: Monodepth2Format, batch: Dict) -> Monodepth2Format:
        """ Returns mask from flow consistency. Parametrized through flow_cons_threshold
        Flow consistency is defined as (flow_tgt2ref_tgt + flow_ref2tgt_tgt) < thresh where the subscript tgt means
        that the respective flow is in target frame coordinates. Consistent flow means that forward + backward flow
        ends up at the same location in the target. This fails for the induced flow between two timesteps on
        - dynamic objects
        - occluded regions
        Args:
            inputs: Dict of network input data in monodepth2 format
            batch: Input data
        Returns:
            masks: Flow consistency masks in monodepth2 format
        """
        masks = {}

        tgt_depth = 1 / batch['others'][0]['disp_up'][0]
        tgt_pose = inputs[('T', 0, 0)]
        tgt_points = self.backproject_depth(tgt_depth, inputs[("inv_K", 0, 0, 0)])
        h, w = tgt_depth.shape[-2:]
        coords0 = self.backproject_depth.id_coords.permute(1, 2, 0)

        for cam in self.contexts:
            for ctx in self.contexts[cam]:
                ref_depth = 1 / batch['others'][ctx]['disp_up'][cam]
                ref_pose = inputs[('T', cam, ctx)]
                px_tgt_ref = self.project_3d(tgt_points, inputs[("K", cam, ctx, 0)], ref_pose.inverse() @ tgt_pose)
                ref_points = self.backproject_depth(ref_depth, inputs[("inv_K", cam, ctx, 0)])
                px_ref_tgt = self.project_3d(ref_points, inputs[("K", 0, 0, 0)], tgt_pose.inverse() @ ref_pose)

                flow_tgt_ref = (px_tgt_ref + 1) * torch.tensor([w - 2, h - 2], device=px_tgt_ref.device) / 2 - coords0
                flow_ref_tgt = (px_ref_tgt + 1) * torch.tensor([w - 2, h - 2], device=px_ref_tgt.device) / 2 - coords0

                delta_flow_tgt = flow_tgt_ref + F.grid_sample(
                    flow_ref_tgt.permute(0, 3, 1, 2),
                    px_tgt_ref,
                    padding_mode="zeros",
                    align_corners=True  # ToDo: Check
                ).permute(0, 2, 3, 1)

                mask = delta_flow_tgt.norm(dim=-1)[:, None] < self.flow_cons_threshold
                if ('mask', cam, ctx, 0) in inputs:
                    mask &= inputs[('mask', cam, ctx, 0)]
                masks.update({('mask', cam, ctx, 0): mask})
        return masks

    def forward(self, batch, **kwargs) -> Dict[str, Union[Dict, DepthOutput]]:
        torch.manual_seed(0)    # ToDo: Debugging
        # --- Prepare Inputs ---
        inputs = {}
        tgt_cam = 0
        batch_size, n_cams, _, height, width = batch['rgb'][0].shape
        if self.training:
            # --- Build input ---
            for ctx in batch['rgb']:
                for cam in range(batch['rgb'][ctx].shape[1]):
                    intr = intr_3x3_to_4x4(batch['intrinsics'][ctx][:, cam])
                    inputs.update({
                        ('color', cam, ctx, 0): batch['rgb'][ctx][:, cam],
                        ('K', cam, ctx, 0): intr,
                        ('inv_K', cam, ctx, 0): intr.inverse(),
                    })
                    if self.use_gt_pose:
                        pose = batch['pose'][0][:, 0].inverse() @ batch['pose'][ctx][:, cam]
                    else:
                        pose = pose_quaternion_to_matrix(batch['others'][0]['pose'][0]).inverse() @ \
                               pose_quaternion_to_matrix(batch['others'][ctx]['pose'][0])
                        if cam != 0:  # Load relative poses between cameras from gt
                            pose = pose @ batch['pose'][ctx][:, 0].inverse() @ batch['pose'][ctx][:, cam]
                    inputs.update({
                        ('T', cam, ctx): pose,
                    })

            # --- Create masks ---
            if 'mask' in batch:
                inputs.update({('mask', tgt_cam, 0, 0): batch['mask'][0][:, tgt_cam] > 0})
            if self.flow_cons_threshold > 0 and 'others' in batch and 'disp_up' in batch['others'][0]:
                inputs.update(self.get_flow_consistency_masks(inputs, batch))
            if self.max_depth_threshold > 0 and 'others' in batch and 'disp_up' in batch['others'][0]:
                inputs.update(self.get_max_depth_masks(inputs, batch))
            if 'others' in batch and 'conf_up' in batch['others'][0]:
                inputs.update({('conf', tgt_cam, 0, 0): batch['others'][0]['conf_up'][tgt_cam]})

            # --- Augment Inputs ---
            # Color augmentation
            inputs.update({('color', tgt_cam, 0, i): self.resize[i](inputs[('color', tgt_cam, 0, 0)])
                           for i in range(1, self.num_scales)})
            inputs.update({('color_aug', tgt_cam, 0, 0): torch.stack([self.col_jit(rgb)
                                                                      for rgb in inputs[('color', tgt_cam, 0, 0)]])})

            inp_intr = inputs[('K', tgt_cam, 0, 0)][:, 0, 0]
            inp_image = inputs[('color_aug', tgt_cam, 0, 0)]
            inp_disps = batch['others'][0]['disp'][tgt_cam]
            inp_conf = batch['others'][0]['conf'][tgt_cam]
            inp_disps_up = batch['others'][0]['disp_up'][tgt_cam]
            inp_conf_up = batch['others'][0]['conf_up'][tgt_cam]

            # Randomly set inputs to 0
            augment_idcs = torch.rand(batch_size) < self.p_zero_inp
            inp_disps[augment_idcs] = 0.0
            inp_disps_up[augment_idcs] = 0.0
            inp_conf[augment_idcs] = 0.0
            inp_conf_up[augment_idcs] = 0.0
        else:
            # --- Prepare Inputs ---
            inp_intr = batch['intrinsics'][0][:, :, 0, 0].view(batch_size * n_cams)
            inp_image = batch['rgb'][0].view(batch_size * n_cams, 3, height, width)
            inp_disps = torch.stack(batch['others'][0]['disp'], 1).view(batch_size * n_cams, 1, height // 8, width // 8)
            inp_conf = torch.stack(batch['others'][0]['conf'], 1).view(batch_size * n_cams, 1, height // 8, width // 8)
            inp_disps_up = torch.stack(batch['others'][0]['disp_up'], 1).view(batch_size * n_cams, 1, height, width)
            inp_conf_up = torch.stack(batch['others'][0]['conf_up'], 1).view(batch_size * n_cams, 1, height, width)

            # Set input to 0 if not given
            no_inp_idcs = inp_conf.view(inp_conf.shape[0], -1).max(dim=1)[0] == 0
            inp_disps[no_inp_idcs] = 0.0
            inp_disps_up[no_inp_idcs] = 0.0

        # Random flipping
        a = torch.rand(1)
        if self.training and self.p_flip > a:
            def flip(x):
                return hflip(x)
        else:
            def flip(x):
                return x

        # --- Inference ---
        outputs = self.networks.completion(
            flip(inp_image),
            flip(inp_disps),
            flip(inp_conf),
            flip(inp_disps_up),
            flip(inp_conf_up),
            inp_intr,
        )

        # flip back predictions
        outputs = {key: flip(outputs[key]) for key in outputs}

        if self.training:
            # --- Loss ---
            self.generate_images_pred(inputs, outputs)
            losses = {'loss': 0}
            for ref_cam in self.contexts:
                automasking = (not self.disable_automasking) and (ref_cam == 0)
                cam_losses = self.compute_losses(inputs, outputs, ref_cam=ref_cam, automasking=automasking)
                weight = self.weight_tgt_cam if ref_cam == 0 else self.weight_ref_cam
                losses['loss'] += weight * cam_losses['loss']
                losses.update({f'cam{ref_cam}_{key}': loss for key, loss in cam_losses.items()})

            return {
                'loss': losses['loss'],
                'metrics': {},
                'predictions': {
                    'depth': {0: outputs[('depth', 0)][:, None]}
                }
            }
        else:
            return {
                'metrics': {},
                'predictions': {
                    'depth': {0: [outputs[('depth', 0)].view(batch_size, n_cams, 1, height, width)]}
                }
            }

    def generate_images_pred(self, inputs: Monodepth2Format, outputs: Monodepth2Format):
        """Generate the warped (reprojected) color images for a minibatch. Generated images are saved into the `outputs`
        dictionary.
        Args:
            inputs: Inference input data in monodepth2 format
            outputs: Inference output data in monodepth2 format
        """
        src_scale = 0
        b, _, h, w = inputs[('color', 0, 0, src_scale)].shape
        for scale in self.scales:
            depth = outputs[("depth", scale)]
            depth = F.interpolate(depth, (h, w), mode="bilinear", align_corners=False)  # get original scale of pred.

            for cam in self.contexts:
                for ctx in self.contexts[cam]:
                    pred2ref = inputs[('T', cam, ctx)].inverse() @ inputs[('T', 0, 0)]

                    cam_points = self.backproject_depth(depth, inputs[("inv_K", 0, 0, src_scale)])
                    pix_coords = self.project_3d(cam_points, inputs[("K", cam, ctx, src_scale)], pred2ref)

                    outputs[("sample", cam, ctx, scale)] = pix_coords

                    outputs[("color", cam, ctx, scale)] = F.grid_sample(
                        inputs[("color", cam, ctx, src_scale)],
                        outputs[("sample", cam, ctx, scale)],
                        padding_mode="border"
                    )

                    if not self.disable_automasking and cam == 0:
                        outputs[("color_identity", cam, ctx, scale)] = inputs[("color", cam, ctx, src_scale)]

    def compute_losses(
            self,
            inputs: Monodepth2Format,
            outputs: Monodepth2Format,
            ref_cam: Optional[int] = 0,
            automasking: Optional[bool] = False
    ) -> Losses:
        """ Compute the view-synthesis and smoothness losses for a minibatch
        Args:
            inputs: Network inputs
            outputs: Predictions in monodepth2 format (including warped images)
            ref_cam: Which camera should be used as reference. Has to be in self.context.keys()
            automasking: If automasking (see monodepth2) should be used
        """
        losses = {}
        total_loss = 0.0

        for scale in self.scales:
            loss = 0
            reprojection_losses = []

            src_scale = 0

            for ctx in self.contexts[ref_cam]:
                reprojection_loss = self.compute_photo_loss(
                    outputs[("color", ref_cam, ctx, scale)],        # prediction (warped)
                    inputs[("color", 0, 0, src_scale)]              # target
                )

                if ('mask', ref_cam, ctx, src_scale) in inputs:
                    reprojection_loss[~inputs[('mask', ref_cam, ctx, src_scale)]] = float("inf")
                if ('mask', 0, 0, src_scale) in inputs and inputs[('mask', 0, 0, src_scale)] is not None:
                    reprojection_loss[~inputs[('mask', 0, 0, src_scale)]] = float("inf")
                reprojection_losses.append(reprojection_loss)

            reprojection_loss = torch.cat(reprojection_losses, 1)

            if automasking:
                identity_reprojection_losses = []
                for ctx in self.contexts[ref_cam]:
                    identity_reprojection_losses.append(
                        self.compute_photo_loss(
                            inputs[("color", ref_cam, ctx, src_scale)],     # not warped
                            inputs[("color", 0, 0, src_scale)])             # target
                    )

                identity_reprojection_loss = torch.cat(identity_reprojection_losses, 1)

                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=identity_reprojection_loss.device
                ) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            to_optimise, idxs = torch.min(combined, dim=1)
            to_optimise[to_optimise.isinf()] = 0
            if self.confidence_thresh > 0.0:
                to_optimise *= (inputs[('conf', 0, 0, src_scale)][:, 0] > self.confidence_thresh)

            loss += to_optimise.mean()

            mean_disp = outputs[("disp", scale)].mean(2, True).mean(3, True)
            norm_disp = outputs[("disp", scale)] / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, inputs[("color", 0, 0, scale)])

            loss += self.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def compute_photo_loss(self, pred: torch.Tensor, target: torch.Tensor, ssim: Optional[bool] = True) -> torch.Tensor:
        """ Computes reprojection loss between a batch of predicted and target images
        Args:
            pred: Warped ref-view -> tgt-view based on predicted depth of shape (B, 3, H, W) where B = batch-size
            target: Target-view of shape (B, 3, H, W)
            ssim: False - only use L1 loss, True - use L1 & SSIM loss, default: True
        Returns:
            loss: Photometric loss based on SSIM & L1 of shape (B, 1, H, W)
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if not ssim:
            photometric_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            photometric_loss = self.weight_ssim * ssim_loss + self.weight_l1 * l1_loss

        return photometric_loss
