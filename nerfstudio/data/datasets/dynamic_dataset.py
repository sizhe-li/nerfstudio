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

from typing import Dict

import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor

# from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.dataparsers.dnerf_dataparser import DNeRFDataParserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import get_depth_image_from_path


class DynamicDataset(InputDataset):
    def __init__(
        self, dataparser_outputs: DNeRFDataParserOutputs, scale_factor: float = 1.0
    ):
        super().__init__(dataparser_outputs, scale_factor)

        self.times = dataparser_outputs.metadata["times"]

    def get_metadata(self, data: Dict) -> Dict:
        ret_dict = dict()

        image_idx = data["image_idx"]
        broad_cast_shape_single_dim = data["image"].shape[:-1] + (1,)  # "* H W 1"

        if self.times is not None:
            times = self.times[image_idx].broadcast_to(broad_cast_shape_single_dim)
            ret_dict["times"] = times

        return ret_dict


class DynamicDepthDataset(DynamicDataset):
    """Dataset that returns images and depths.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs.
    """

    # exclude_batch_keys_from_device = InputDataset.exclude_batch_keys_from_device + [
    #     "depth_image"
    # ]

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
        metadata = super().get_metadata(data)

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

        metadata["depth_image"] = depth_image
        return metadata


class DynamicDepthFeatureDataset(DynamicDepthDataset):
    """Dataset that returns images and depths.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs.
    """

    # exclude_batch_keys_from_device = InputDataset.exclude_batch_keys_from_device + [
    #     "dense_features_gt"
    # ]

    def __init__(
        self, dataparser_outputs: DNeRFDataParserOutputs, scale_factor: float = 1.0
    ):
        super().__init__(dataparser_outputs, scale_factor)
        assert (
            "feature_filenames" in dataparser_outputs.metadata.keys()
            and dataparser_outputs.metadata["feature_filenames"] is not None
        )
        self.feature_filenames = self.metadata["feature_filenames"]

    def get_numpy_feature_image(self, image_filename) -> npt.NDArray[np.uint8]:
        """Returns the image of shape (H, W, 3 or 4).

        Args:
            image_idx: The image index in the dataset.
        """
        pil_image = Image.open(image_filename)
        if self.scale_factor != 1.0:
            width, height = pil_image.size
            newsize = (int(width * self.scale_factor), int(height * self.scale_factor))
            pil_image = pil_image.resize(newsize, resample=Image.Resampling.BILINEAR)
        image = np.array(pil_image, dtype="uint8")  # shape is (h, w) or (h, w, 3 or 4)

        if len(image.shape) == 2:
            image = image[:, :, None].repeat(3, axis=2)
        assert len(image.shape) == 3
        assert image.dtype == np.uint8
        assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is in correct."
        return image

    def get_feature_image_float32(
        self,
        image_filename,
    ) -> Float[Tensor, "image_height image_width num_channels"]:
        """Returns a 3 channel image in float32 torch.Tensor.

        Args:
            image_idx: The image index in the dataset.
        """
        image = torch.from_numpy(
            self.get_numpy_feature_image(image_filename).astype("float32") / 255.0
        )
        if self._dataparser_outputs.alpha_color is not None and image.shape[-1] == 4:
            assert (self._dataparser_outputs.alpha_color >= 0).all() and (
                self._dataparser_outputs.alpha_color <= 1
            ).all(), "alpha color given is out of range between [0, 1]."
            image = image[:, :, :3] * image[
                :, :, -1:
            ] + self._dataparser_outputs.alpha_color * (1.0 - image[:, :, -1:])
        return image

    def get_metadata(self, data: Dict) -> Dict:
        metadata = super().get_metadata(data)

        filepath = self.feature_filenames[data["image_idx"]]
        feature_gt = self.get_feature_image_float32(filepath)

        # feature_gt = data["image"]

        metadata["feature_gt"] = feature_gt

        return metadata
