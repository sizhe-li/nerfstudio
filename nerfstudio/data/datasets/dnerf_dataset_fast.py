# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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
Depth dataset.
"""

import torch
import numpy as np
from typing import Dict

from nerfstudio.data.dataparsers.dnerf_dataparser import DNeRFDataParserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import get_depth_image_from_path
from nerfstudio.data.pixel_samplers import PixelSampler


class DynamicDatasetFast(InputDataset):
    def __init__(
        self, dataparser_outputs: DNeRFDataParserOutputs, scale_factor: float = 1.0
    ):
        super().__init__(dataparser_outputs, scale_factor)

        self.times = dataparser_outputs.metadata["times"]
        self.sample_inds = dataparser_outputs.metadata["sample_inds"]
        self.num_rays = 128
        # self.pixel_sampler = PixelSampler(num_rays_per_batch=1024)
        self.image_coords = self.cameras.get_image_coords()

    def get_metadata(self, data: Dict) -> Dict:
        ret_dict = dict()

        image_idx = data["image_idx"]
        broad_cast_shape = data["image"].shape[:-1] + (1,)  # "* H W 1"

        if self.times is not None:
            times = self.times[image_idx].broadcast_to(broad_cast_shape)
            ret_dict["times"] = times
        if self.sample_inds is not None:
            sample_inds = self.sample_inds[image_idx].broadcast_to(broad_cast_shape)
            ret_dict["sample_inds"] = sample_inds

        return ret_dict

    def __getitem__(self, image_idx: int) -> Dict:
        batch = self.get_data(image_idx)

        # sample pixels
        image_height, image_width = self.image_coords.shape[:2]
        indices = torch.floor(
            torch.rand((self.num_rays, 3))
            * torch.tensor([0, image_height, image_width])
        ).long()

        _, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))

        for k, v in batch.items():
            if k != "image_idx" and v is not None:
                batch[k] = v[y, x]

        batch.pop("image_idx")

        indices[:, 0] = self._dataparser_outputs.sample_to_camera_idx[image_idx].item()
        batch["indices"] = indices

        # camera_idx = self._dataparser_outputs.sample_to_camera_idx[image_idx].item()

        # sample_inds = batch["sample_inds"]
        # coords = self.image_coords[y, x]

        # ray_bundle = self.cameras.generate_rays(
        #     camera_indices=camera_idx,
        #     coords=coords,
        #     sample_inds=sample_inds,
        # )

        return batch


class DynamicDepthDatasetFast(DynamicDatasetFast):
    """Dataset that returns images and depths.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs.
    """

    exclude_batch_keys_from_device = InputDataset.exclude_batch_keys_from_device + [
        "depth_image"
    ]

    def __init__(
        self, dataparser_outputs: DNeRFDataParserOutputs, scale_factor: float = 1.0
    ):
        super().__init__(dataparser_outputs, scale_factor)
        assert (
            "depth_filenames" in dataparser_outputs.metadata.keys()
            and dataparser_outputs.metadata["depth_filenames"] is not None
        )
        self.depth_filenames = self.metadata["depth_filenames"]
        self.depth_unit_scale_factor = self.metadata["depth_unit_scale_factor"]

    def get_metadata(self, data: Dict) -> Dict:
        ret_dict = super().get_metadata(data)

        filepath = self.depth_filenames[data["image_idx"]]

        self._dataparser_outputs: DNeRFDataParserOutputs
        cam_idx = self._dataparser_outputs.sample_to_camera_idx[data["image_idx"]]
        height = int(self._dataparser_outputs.cameras.height[cam_idx])
        width = int(self._dataparser_outputs.cameras.width[cam_idx])

        # Scale depth images to meter units and also by scaling applied to cameras
        scale_factor = (
            self.depth_unit_scale_factor * self._dataparser_outputs.dataparser_scale
        )
        depth_image = get_depth_image_from_path(
            filepath=filepath, height=height, width=width, scale_factor=scale_factor
        )

        ret_dict["depth_image"] = depth_image
        return ret_dict


class DynamicDepthFeatureDatasetFast(DynamicDepthDatasetFast):
    """Dataset that returns images and depths.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs.
    """

    exclude_batch_keys_from_device = InputDataset.exclude_batch_keys_from_device + [
        "dense_features_gt"
    ]

    def __init__(
        self, dataparser_outputs: DNeRFDataParserOutputs, scale_factor: float = 1.0
    ):
        super().__init__(dataparser_outputs, scale_factor)
        assert (
            "dense_features_filenames" in dataparser_outputs.metadata.keys()
            and dataparser_outputs.metadata["dense_features_filenames"] is not None
        )
        self.dense_features_filenames = self.metadata["dense_features_filenames"]

    def get_metadata(self, data: Dict) -> Dict:
        ret_dict = super().get_metadata(data)

        filepath = self.dense_features_filenames[data["image_idx"]]
        ret_dict["dense_features_gt"] = torch.from_numpy(
            np.load(filepath)["dense_features_gt"]
        )

        # self._dataparser_outputs: DNeRFDataParserOutputs
        # cam_idx = self._dataparser_outputs.sample_to_camera_idx[data["image_idx"]]
        # height = int(self._dataparser_outputs.cameras.height[cam_idx])
        # width = int(self._dataparser_outputs.cameras.width[cam_idx])

        # # Scale depth images to meter units and also by scaling applied to cameras
        # scale_factor = (
        #     self.depth_unit_scale_factor * self._dataparser_outputs.dataparser_scale
        # )
        # depth_image = get_depth_image_from_path(
        #     filepath=filepath, height=height, width=width, scale_factor=scale_factor
        # )

        # ret_dict["depth_image"] = depth_image
        return ret_dict
