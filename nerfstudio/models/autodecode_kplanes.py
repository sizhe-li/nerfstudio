# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Implementation of K-Planes (https://sarafridov.github.io/K-Planes/).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

import math
import numpy as np
import torch
import torch.nn as nn
import nerfacc
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.losses import MSELoss, distortion_loss, interlevel_loss
from nerfstudio.model_components.ray_samplers import (
    ProposalNetworkSampler,
    UniformLinDispPiecewiseSampler,
    UniformSampler,
)
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
    FeatureRenderer,
)
from nerfstudio.model_components.autodecode_ray_samplers import (
    AutoDecodeVolumetricSampler,
    DensityFn,
)
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, misc

from nerfstudio.fields.autodecode_kplanes_field import KPlanesDensityField, KPlanesField


@dataclass
class KPlanesModelConfig(ModelConfig):
    """K-Planes Model Config"""

    _target: Type = field(default_factory=lambda: KPlanesModel)

    near_plane: float = 2.0
    """How far along the ray to start sampling."""

    far_plane: float = 6.0
    """How far along the ray to stop sampling."""

    grid_base_resolution: List[int] = field(default_factory=lambda: [128, 128, 128])
    """Base grid resolution."""

    grid_feature_dim: int = 32
    """Dimension of feature vectors stored in grid."""

    multiscale_res: List[int] = field(default_factory=lambda: [1, 2, 4])
    """Multiscale grid resolutions."""

    is_contracted: bool = False
    """Whether to use scene contraction (set to true for unbounded scenes)."""

    concat_features_across_scales: bool = True
    """Whether to concatenate features at different scales."""

    linear_decoder: bool = False
    """Whether to use a linear decoder instead of an MLP."""

    linear_decoder_layers: Optional[int] = 1
    """Number of layers in linear decoder"""

    # proposal sampling arguments
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""

    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""

    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"num_output_coords": 8, "resolution": [64, 64, 64]},
            {"num_output_coords": 8, "resolution": [128, 128, 128]},
        ]
    )
    """Arguments for the proposal density fields."""

    num_proposal_samples: Optional[Tuple[int]] = (256, 128)
    """Number of samples per ray for each proposal network."""

    num_samples: Optional[int] = 48
    """Number of samples per ray used for rendering."""

    single_jitter: bool = False
    """Whether use single jitter or not for the proposal networks."""

    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps."""

    proposal_update_every: int = 5
    """Sample every n steps after the warmup."""

    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""

    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""

    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""

    appearance_embedding_dim: int = 0
    """Dimension of appearance embedding. Set to 0 to disable."""

    use_average_appearance_embedding: bool = True
    """Whether to use average appearance embedding or zeros for inference."""

    background_color: Literal["random", "last_sample", "black", "white"] = "white"
    """The background color as RGB."""

    loss_coefficients: Dict[str, float] = to_immutable_dict(
        {
            "img": 1.0,
            # "dense_features": 0.001,
            "interlevel": 1.0,
            "distortion": 0.001,
            "plane_tv": 0.0001,
            "plane_tv_proposal_net": 0.0001,
            "l1_codes": 0.0001,
            "codes_smoothness": 0.0001,
        }
    )
    """Loss coefficients."""

    use_occupancy_grid: bool = False
    """ Whether to use occupancy grid for rendering. """
    occ_grid_resolution: int = 128
    """Resolution of the grid used for the field."""
    occ_grid_levels: int = 1
    """Levels of the grid used for the field."""
    occ_step_size: Optional[float] = None
    """Minimum step size for rendering."""
    alpha_thre: float = 0.01
    """Threshold for opacity skipping."""
    cone_angle: float = 0.004
    """Should be set to 0.0 for blender scenes but 1./256 for real scenes."""


class KPlanesModel(Model):
    config: KPlanesModelConfig
    """K-Planes model

    Args:
        config: K-Planes configuration to instantiate model
    """

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.is_contracted:
            scene_contraction = SceneContraction(order=float("inf"))
        else:
            scene_contraction = None

        self.n_samples = self.kwargs["metadata"]["sample_inds"].max() + 1

        self.sample_embedding = nn.Embedding(
            self.n_samples, self.config.grid_feature_dim
        )
        torch.nn.init.normal_(
            self.sample_embedding.weight,
            mean=0.0,
            std=0.01 / math.sqrt(self.config.grid_feature_dim),
        )

        # Fields
        self.field = KPlanesField(
            self.scene_box.aabb,
            num_images=self.num_train_data,
            grid_base_resolution=self.config.grid_base_resolution,
            grid_feature_dim=self.config.grid_feature_dim,
            sample_code_dim=self.config.grid_feature_dim,
            concat_across_scales=self.config.concat_features_across_scales,
            multiscale_res=self.config.multiscale_res,
            spatial_distortion=scene_contraction,
            appearance_embedding_dim=self.config.appearance_embedding_dim,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            linear_decoder=self.config.linear_decoder,
            linear_decoder_layers=self.config.linear_decoder_layers,
        )

        if self.config.use_occupancy_grid:
            self.scene_aabb = Parameter(
                self.scene_box.aabb.flatten(), requires_grad=False
            )

            if self.config.occ_step_size is None:
                # auto step size: ~1000 samples in the base level grid
                self.config.occ_step_size = (
                    (self.scene_aabb[3:] - self.scene_aabb[:3]) ** 2
                ).sum().sqrt().item() / 1000

            self.occupancy_grid = nerfacc.OccGridEstimator(
                roi_aabb=self.scene_aabb,
                resolution=self.config.occ_grid_resolution,
                levels=self.config.occ_grid_levels,
            )

            self.sampler = AutoDecodeVolumetricSampler(
                occupancy_grid=self.occupancy_grid,
                density_fn=self.field.density_fn,
            )
        else:
            self.density_fns = []
            num_prop_nets = self.config.num_proposal_iterations
            # Build the proposal network(s)
            self.proposal_networks = torch.nn.ModuleList()
            if self.config.use_same_proposal_network:
                assert (
                    len(self.config.proposal_net_args_list) == 1
                ), "Only one proposal network is allowed."
                prop_net_args = self.config.proposal_net_args_list[0]
                network = KPlanesDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    linear_decoder=self.config.linear_decoder,
                    sample_code_dim=self.config.grid_feature_dim,
                    **prop_net_args,
                )
                self.proposal_networks.append(network)
                self.density_fns.extend(
                    [network.density_fn for _ in range(num_prop_nets)]
                )
            else:
                for i in range(num_prop_nets):
                    prop_net_args = self.config.proposal_net_args_list[
                        min(i, len(self.config.proposal_net_args_list) - 1)
                    ]
                    network = KPlanesDensityField(
                        self.scene_box.aabb,
                        spatial_distortion=scene_contraction,
                        linear_decoder=self.config.linear_decoder,
                        sample_code_dim=self.config.grid_feature_dim,
                        **prop_net_args,
                    )
                    self.proposal_networks.append(network)
                self.density_fns.extend(
                    [network.density_fn for network in self.proposal_networks]
                )

            # Samplers
            def update_schedule(step):
                return np.clip(
                    np.interp(
                        step,
                        [0, self.config.proposal_warmup],
                        [0, self.config.proposal_update_every],
                    ),
                    1,
                    self.config.proposal_update_every,
                )

            if self.config.is_contracted:
                initial_sampler = UniformLinDispPiecewiseSampler(
                    single_jitter=self.config.single_jitter
                )
            else:
                initial_sampler = UniformSampler(
                    single_jitter=self.config.single_jitter
                )

            self.proposal_sampler = ProposalNetworkSampler(
                num_nerf_samples_per_ray=self.config.num_samples,
                num_proposal_samples_per_ray=self.config.num_proposal_samples,
                num_proposal_network_iterations=self.config.num_proposal_iterations,
                single_jitter=self.config.single_jitter,
                update_sched=update_schedule,
                initial_sampler=initial_sampler,
            )

        # Collider
        self.collider = NearFarCollider(
            near_plane=self.config.near_plane, far_plane=self.config.far_plane
        )

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")
        self.renderer_features = FeatureRenderer()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.temporal_distortion = (
            len(self.config.grid_base_resolution) == 4
        )  # for viewer

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {
            "fields": list(self.field.parameters()),
            "embeddings": list(self.sample_embedding.parameters()),
        }
        if not self.config.use_occupancy_grid:
            param_groups["proposal_networks"] = list(
                self.proposal_networks.parameters()
            )

        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_occupancy_grid:

            def update_occupancy_grid(step: int):
                def occ_eval_fn(x):
                    sample_inds = torch.randint(
                        0, self.n_samples, (x.shape[0],), device=x.device
                    )
                    conditioning_codes = self.sample_embedding(sample_inds)
                    density = self.config.occ_step_size * self.field.density_fn(
                        x, conditioning_codes=conditioning_codes
                    )
                    return density

                self.occupancy_grid.update_every_n_steps(
                    step=step,
                    occ_eval_fn=occ_eval_fn,
                    ema_decay=0.99,
                )

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=update_occupancy_grid,
                )
            )
        else:
            if self.config.use_proposal_weight_anneal:
                # anneal the weights of the proposal network before doing PDF sampling
                N = self.config.proposal_weights_anneal_max_num_iters

                def set_anneal(step):
                    # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                    train_frac = np.clip(step / N, 0, 1)
                    bias = lambda x, b: (b * x) / ((b - 1) * x + 1)
                    anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                    self.proposal_sampler.set_anneal(anneal)

                callbacks.append(
                    TrainingCallback(
                        where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                        update_every_num_iters=1,
                        func=set_anneal,
                    )
                )
                callbacks.append(
                    TrainingCallback(
                        where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                        update_every_num_iters=1,
                        func=self.proposal_sampler.step_cb,
                    )
                )

        return callbacks

    def get_outputs(self, ray_bundle: RayBundle):
        conditioning_codes = self.sample_embedding(ray_bundle.sample_inds)
        if not self.training:
            if "conditioning_codes" in ray_bundle.metadata:
                conditioning_codes = ray_bundle.metadata["conditioning_codes"]

        num_rays = len(ray_bundle)

        weights_list = []
        ray_samples_list = []
        if self.config.use_occupancy_grid:
            ray_bundle.metadata["conditioning_codes"] = conditioning_codes
            with torch.no_grad():
                ray_samples, ray_indices = self.sampler(
                    ray_bundle=ray_bundle,
                    near_plane=self.config.near_plane,
                    far_plane=self.config.far_plane,
                    render_step_size=self.config.occ_step_size,
                    alpha_thre=self.config.alpha_thre,
                    cone_angle=self.config.cone_angle,
                )
                packed_info = nerfacc.pack_info(ray_indices, num_rays)
                field_outputs = self.field(ray_samples)

        else:
            density_fns = self.density_fns
            density_fns = [
                functools.partial(f, conditioning_codes=conditioning_codes)
                for f in density_fns
            ]

            ray_samples: RaySamples
            ray_samples, weights_list, ray_samples_list = self.proposal_sampler(
                ray_bundle, density_fns=density_fns
            )

            ray_samples.metadata["conditioning_codes"] = conditioning_codes
            field_outputs = self.field(ray_samples)

            ray_indices = torch.arange(num_rays, device=ray_bundle.origins.device)
            ray_indices = torch.repeat_interleave(ray_indices, ray_samples.shape[1])
            ray_samples.frustums.starts = ray_samples.frustums.starts.reshape(-1, 1)
            ray_samples.frustums.ends = ray_samples.frustums.ends.reshape(-1, 1)
            packed_info = nerfacc.pack_info(ray_indices, num_rays)

        weights = nerfacc.render_weight_from_density(
            t_starts=ray_samples.frustums.starts[..., 0],
            t_ends=ray_samples.frustums.ends[..., 0],
            sigmas=field_outputs[FieldHeadNames.DENSITY][..., 0].view(-1),
            packed_info=packed_info,
        )[0]
        weights = weights[..., None]

        rgb = self.renderer_rgb(
            rgb=field_outputs[FieldHeadNames.RGB].view(-1, 3),
            weights=weights,
            ray_indices=ray_indices,
            num_rays=num_rays,
        )
        depth = self.renderer_depth(
            weights=weights,
            ray_samples=ray_samples,
            ray_indices=ray_indices,
            num_rays=num_rays,
        )

        if not self.config.use_occupancy_grid:
            weights_list.append(weights.reshape(num_rays, -1, 1))
            ray_samples_list.append(ray_samples)
            ray_samples.frustums.starts = ray_samples.frustums.starts.reshape(
                num_rays, -1, 1
            )
            ray_samples.frustums.ends = ray_samples.frustums.ends.reshape(
                num_rays, -1, 1
            )

            # ndf_features = self.renderer_features(
            #     features=field_outputs["ndf_features"],
            #     weights=weights_list[-1],
            # )
            # steps = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2
        outputs = {
            "rgb": rgb,
            "depth": depth,
            "weights": (weights,),
            # "steps": (steps, )
            # "ndf_features": (ndf_features, )
        }

        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            if self.config.use_occupancy_grid:
                weights_list.append(weights)
                ray_samples_list.append(ray_samples)

            outputs["weights_list"] = (weights_list,)
            outputs["ray_samples_list"] = (ray_samples_list,)

        # for i in range(self.config.num_proposal_iterations):
        #     outputs[f"prop_depth_{i}"] = self.renderer_depth(
        #         weights=weights_list[i], ray_samples=ray_samples_list[i]
        #     )

        # prop_grids = [p.grids.plane_coefs for p in self.proposal_networks]
        field_grids = [g.plane_coefs for g in self.field.grids]

        outputs["plane_tv"] = (space_tv_loss(field_grids),)

        return outputs

    def get_metrics_dict(self, outputs, batch):
        image = batch["image"].to(self.device)

        metrics_dict = {"psnr": self.psnr(outputs["rgb"], image)}
        if self.training:
            weights_list = outputs["weights_list"][0]
            ray_sample_list = outputs["ray_samples_list"][0]

            metrics_dict["interlevel"] = interlevel_loss(
                weights_list, ray_sample_list,
            )
            metrics_dict["distortion"] = distortion_loss(
                weights_list, ray_sample_list,
            )

            metrics_dict["plane_tv"] = outputs["plane_tv"][0]

        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        image = batch["image"].to(self.device)
        # dense_features_gt = batch["dense_features_gt"].to(self.device)

        loss_dict = {
            "rgb": self.rgb_loss(image, outputs["rgb"]),
            # "dense_features": self.rgb_loss(
            #     dense_features_gt, outputs["dense_features"]
            # ),
        }
        if self.training:
            for key in self.config.loss_coefficients:
                if key in metrics_dict:
                    loss_dict[key] = metrics_dict[key].clone()

            loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)

        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(self.device)

        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"], accumulation=outputs["accumulation"]
        )

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        # all of these metrics will be logged as scalars
        metrics_dict = {
            "psnr": float(self.psnr(image, rgb).item()),
            "ssim": float(self.ssim(image, rgb)),
            "lpips": float(self.lpips(image, rgb)),
        }
        images_dict = {
            "img": combined_rgb,
            "accumulation": combined_acc,
            "depth": combined_depth,
        }

        # for i in range(self.config.num_proposal_iterations):
        #     key = f"prop_depth_{i}"
        #     prop_depth_i = colormaps.apply_depth_colormap(
        #         outputs[key],
        #         accumulation=outputs["accumulation"],
        #     )
        #     images_dict[key] = prop_depth_i

        return metrics_dict, images_dict


def compute_plane_tv(t: torch.Tensor, only_w: bool = False) -> float:
    """Computes total variance across a plane.

    Args:
        t: Plane tensor
        only_w: Whether to only compute total variance across w dimension

    Returns:
        Total variance
    """
    _, h, w = t.shape
    w_tv = torch.square(t[..., :, 1:] - t[..., :, : w - 1]).mean()

    if only_w:
        return w_tv

    h_tv = torch.square(t[..., 1:, :] - t[..., : h - 1, :]).mean()
    return h_tv + w_tv


def space_tv_loss(multi_res_grids: List[torch.Tensor]) -> float:
    """Computes total variance across each spatial plane in the grids.

    Args:
        multi_res_grids: Grids to compute total variance over

    Returns:
        Total variance
    """

    total = 0.0
    num_planes = 0
    for grids in multi_res_grids:
        if len(grids) == 3:
            spatial_planes = {0, 1, 2}
        else:
            spatial_planes = {0, 1, 3}

        for grid_id, grid in enumerate(grids):
            if grid_id in spatial_planes:
                total += compute_plane_tv(grid)
            else:
                # Space is the last dimension for space-time planes.
                total += compute_plane_tv(grid, only_w=True)
            num_planes += 1
    return total / num_planes


def l1_time_planes(multi_res_grids: List[torch.Tensor]) -> float:
    """Computes the L1 distance from the multiplicative identity (1) for spatiotemporal planes.

    Args:
        multi_res_grids: Grids to compute L1 distance over

    Returns:
         L1 distance from the multiplicative identity (1)
    """
    time_planes = [2, 4, 5]  # These are the spatiotemporal planes
    total = 0.0
    num_planes = 0
    for grids in multi_res_grids:
        for grid_id in time_planes:
            total += torch.abs(1 - grids[grid_id]).mean()
            num_planes += 1

    return total / num_planes


def compute_plane_smoothness(t: torch.Tensor) -> float:
    """Computes smoothness across the temporal axis of a plane

    Args:
        t: Plane tensor

    Returns:
        Time smoothness
    """
    _, h, _ = t.shape
    # Convolve with a second derivative filter, in the time dimension which is dimension 2
    first_difference = t[..., 1:, :] - t[..., : h - 1, :]  # [c, h-1, w]
    second_difference = (
        first_difference[..., 1:, :] - first_difference[..., : h - 2, :]
    )  # [c, h-2, w]
    # Take the L2 norm of the result
    return torch.square(second_difference).mean()


def time_smoothness(multi_res_grids: List[torch.Tensor]) -> float:
    """Computes smoothness across each time plane in the grids.

    Args:
        multi_res_grids: Grids to compute time smoothness over

    Returns:
        Time smoothness
    """
    total = 0.0
    num_planes = 0
    for grids in multi_res_grids:
        time_planes = [2, 4, 5]  # These are the spatiotemporal planes
        for grid_id in time_planes:
            total += compute_plane_smoothness(grids[grid_id])
            num_planes += 1

    return total / num_planes
