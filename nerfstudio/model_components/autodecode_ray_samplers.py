from typing import Optional, Tuple

import torch
from jaxtyping import Float
from torch import Tensor

from nerfstudio.cameras.rays import RayBundle, RaySamples

from .ray_samplers import (
    DensityFn,
    Float,
    OccGridEstimator,
    Optional,
    Tensor,
    VolumetricSampler,
    Frustums
)


class AutoDecodeVolumetricSampler(VolumetricSampler):
    def forward(
        self,
        ray_bundle: RayBundle,
        render_step_size: float,
        near_plane: float = 0.0,
        far_plane: Optional[float] = None,
        alpha_thre: float = 0.01,
        cone_angle: float = 0.0,
    ) -> Tuple[RaySamples, Float[Tensor, "total_samples"]]:
        """Generate ray samples in a bounding box.

        Args:
            ray_bundle: Rays to generate samples for
            render_step_size: Minimum step size to use for rendering
            near_plane: Near plane for raymarching
            far_plane: Far plane for raymarching
            alpha_thre: Opacity threshold skipping samples.
            cone_angle: Cone angle for raymarching, set to 0 for uniform marching.

        Returns:
            a tuple of (ray_samples, packed_info, ray_indices)
            The ray_samples are packed, only storing the valid samples.
            The ray_indices contains the indices of the rays that each sample belongs to.
        """

        rays_o = ray_bundle.origins.contiguous()
        rays_d = ray_bundle.directions.contiguous()
        conditioning_codes = ray_bundle.metadata["conditioning_codes"]

        if ray_bundle.nears is not None and ray_bundle.fars is not None:
            t_min = ray_bundle.nears.contiguous().reshape(-1)
            t_max = ray_bundle.fars.contiguous().reshape(-1)

        else:
            t_min = None
            t_max = None

        if far_plane is None:
            far_plane = 1e10

        if ray_bundle.camera_indices is not None:
            camera_indices = ray_bundle.camera_indices.contiguous()
        else:
            camera_indices = None
        ray_indices, starts, ends = self.occupancy_grid.sampling(
            rays_o=rays_o,
            rays_d=rays_d,
            t_min=t_min,
            t_max=t_max,
            sigma_fn=self.get_sigma_fn(rays_o, rays_d, conditioning_codes),
            render_step_size=render_step_size,
            near_plane=near_plane,
            far_plane=far_plane,
            stratified=self.training,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
        )
        num_samples = starts.shape[0]
        if num_samples == 0:
            # create a single fake sample and update packed_info accordingly
            # this says the last ray in packed_info has 1 sample, which starts and ends at 1
            ray_indices = torch.zeros((1,), dtype=torch.long, device=rays_o.device)
            starts = torch.ones((1,), dtype=starts.dtype, device=rays_o.device)
            ends = torch.ones((1,), dtype=ends.dtype, device=rays_o.device)

        # if self.training:
        #     max_num_samples = 3024
        #     if len(ray_indices) > max_num_samples:
        #         reduce_inds = torch.randperm(len(ray_indices))[:max_num_samples]
        #         ray_indices = ray_indices[reduce_inds]
        #         starts = starts[reduce_inds]
        #         ends = ends[reduce_inds]

        origins = rays_o[ray_indices]
        dirs = rays_d[ray_indices]
        if camera_indices is not None:
            camera_indices = camera_indices[ray_indices]


        zeros = torch.zeros_like(origins[:, :1])
        ray_samples = RaySamples(
            frustums=Frustums(
                origins=origins,
                directions=dirs,
                starts=starts[..., None],
                ends=ends[..., None],
                pixel_area=zeros,
            ),
            camera_indices=camera_indices,
        )
        if ray_bundle.metadata.get("conditioning_codes") is not None:
            if not isinstance(ray_samples.metadata, dict):
                ray_samples.metadata = {"conditioning_codes": conditioning_codes[ray_indices]}
            else:
                ray_samples.metadata["conditioning_codes"] = conditioning_codes[ray_indices]

        return ray_samples, ray_indices
