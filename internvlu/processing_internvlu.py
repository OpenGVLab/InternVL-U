# --------------------------------------------------------
# InternVL-U
# Modifications Copyright (c) 2026 OpenGVLab
# This file includes code from Qwen-Image and HuggingFace,
# licensed under the Apache License, Version 2.0.
# --------------------------------------------------------
# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

import math

from typing import Optional, Union, List, Literal, Dict
from enum import Enum, auto

import torch
import numpy as np
import torchvision.transforms as T

from einops import rearrange
from transformers import AutoTokenizer
from transformers.processing_utils import Unpack, ProcessorMixin
from transformers.models.internvl.processing_internvl import (
    InternVLProcessorKwargs,
    InternVLImagesKwargs,
)
from transformers.models.internvl.video_processing_internvl import (
    InternVLVideoProcessor,
)
from transformers.models.got_ocr2 import GotOcr2ImageProcessorFast
from transformers.tokenization_utils_base import TextInput
from transformers.image_transforms import (
    convert_to_rgb,
    resize,
    to_channel_dimension_format,
    pad,
)
from transformers.image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    concatenate_list,
    make_flat_list_of_images,
    make_list_of_images,
    to_numpy_array,
    is_scaled_image,
    is_valid_image,
    infer_channel_dimension_format,
    valid_images,
    validate_preprocess_arguments,
)
from transformers.utils.constants import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
)
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.utils import TensorType, logging
from .vlm.constants import IMG_CONTEXT_TOKEN, IMG_END_TOKEN, IMG_START_TOKEN
from .vlm.conversation import get_conv_template

logger = logging.get_logger(__name__)

RATIO = {
    "any_11ratio": [
        (16, 9),
        (9, 16),
        (7, 5),
        (5, 7),
        (5, 4),
        (4, 5),
        (4, 3),
        (3, 4),
        (3, 2),
        (2, 3),
        (1, 1),
    ],
    "any_9ratio": [
        (16, 9),
        (9, 16),
        (5, 4),
        (4, 5),
        (4, 3),
        (3, 4),
        (3, 2),
        (2, 3),
        (1, 1),
    ],
    "any_7ratio": [(16, 9), (9, 16), (4, 3), (3, 4), (3, 2), (2, 3), (1, 1)],
    "any_5ratio": [(16, 9), (9, 16), (4, 3), (3, 4), (1, 1)],
    "any_1ratio": [(1, 1)],
}

DEFAULT_GEN_RESOLUTION = (512, 512)


class InternVLGenProcessorKwargs(InternVLProcessorKwargs, total=False):
    """Keyword defaults for InternVLU processor components."""

    images_kwargs: InternVLImagesKwargs
    _defaults = {
        "text_kwargs": {
            "padding_side": "left",
        },
        "images_kwargs": {
            "crop_to_patches": False,
        },
        "videos_kwargs": {},
    }


class DropCondType(Enum):
    """Specifies which conditioning sources to drop for generation."""

    none = auto()
    text = auto()
    all = auto()


def pad_images(
    pixel_values_gen_all: torch.Tensor,
    image_grid_thw_gen_all: torch.Tensor,
    merge_size: int = 2,
):
    """Pad a batch of image patch tensors to a uniform grid size.

    Args:
        pixel_values_gen_all (`torch.Tensor`): Flattened patch tokens shaped as
            `(sum(num_patches), channels, patch_height, patch_width)`.
        image_grid_thw_gen_all (`torch.Tensor`): Grid sizes for each image, shaped `(num_images, 3)`
            where each entry is `(t, h, w)` in patch units.
        merge_size (`int`, *optional*, defaults to 2): Merge size used when unflattening patches.

    Returns:
        `torch.Tensor`: Padded image tensors of shape
        `(num_images, channels, max_height * patch_size, max_width * patch_size)`.
    """
    image_grid_thw_gen_max = image_grid_thw_gen_all.max(dim=0).values
    patch_size = pixel_values_gen_all.shape[-1]
    num_images = image_grid_thw_gen_all.shape[0]
    image_num_patches_cum = image_grid_thw_gen_all.prod(dim=-1).cumsum(0)
    image_num_patches_cum = torch.cat(
        [image_num_patches_cum[:1] * 0, image_num_patches_cum], dim=0
    )
    pixel_values_gen_padded = torch.zeros(
        num_images,
        pixel_values_gen_all.shape[1],
        image_grid_thw_gen_max[1] * patch_size,
        image_grid_thw_gen_max[2] * patch_size,
    )
    for image_idx in range(num_images):
        img = pixel_values_gen_all[
            image_num_patches_cum[image_idx] : image_num_patches_cum[image_idx + 1]
        ]
        grid = image_grid_thw_gen_all[image_idx]
        img = rearrange(
            img,
            "(h w p1 p2) c ph pw -> c (h p1 ph) (w p2 pw)",
            h=grid[1] // merge_size,
            w=grid[2] // merge_size,
            p1=merge_size,
        )
        pixel_values_gen_padded[image_idx, :, : img.shape[1], : img.shape[2]] = img
    return pixel_values_gen_padded


def make_batched_images(images) -> List[List[ImageInput]]:
    """
    Accepts images in list or nested list format, and makes a list of images for preprocessing.

    Args:
        images (`Union[List[List[ImageInput]], List[ImageInput], ImageInput]`):
            The input image.

    Returns:
        list: A list of images.
    """
    if (
        isinstance(images, (list, tuple))
        and isinstance(images[0], (list, tuple))
        and is_valid_image(images[0][0])
    ):
        return [img for img_list in images for img in img_list]

    elif isinstance(images, (list, tuple)) and is_valid_image(images[0]):
        return images

    elif is_valid_image(images):
        return [images]

    raise ValueError(f"Could not make batched images from {images}")


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
    max_length: int = None,
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    4. If max_length is not None, the larger dimension does not exceed max_length.

    """
    if height < factor or width < factor:
        raise ValueError(
            f"height:{height} or width:{width} must be larger than factor:{factor}"
        )
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )

    # If max_length is set, scale down to keep the longer side within max_length.
    if max_length is not None and max(height, width) > max_length:
        scale = max_length / max(height, width)
        height = int(height * scale)
        width = int(width * scale)

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    # Re-check max_length since rounding can overshoot the bound.
    if max_length is not None and max(h_bar, w_bar) > max_length:
        scale = max_length / max(h_bar, w_bar)
        h_bar = math.floor(h_bar * scale / factor) * factor
        w_bar = math.floor(w_bar * scale / factor) * factor

    return h_bar, w_bar


def dynamic_resize(h, w, anyres="any_1ratio", anchor_pixels=1024 * 1024, stride=32):
    """Compute a stride-aligned resize target for any-resolution generation.

    Args:
        h (`int`): Original height.
        w (`int`): Original width.
        anyres (`str`, *optional*, defaults to `"any_1ratio"`): Aspect ratio bucket key.
        anchor_pixels (`int`, *optional*, defaults to `1024 * 1024`): Target pixel area used to scale.
        stride (`int`, *optional*, defaults to 32): Minimum spatial alignment stride.

    Returns:
        `Tuple[int, int]`: The resized `(height, width)` in pixels.
    """
    orig_ratio = w / h

    # Choose the closest aspect ratio bucket.
    target_ratio = min(RATIO[anyres], key=lambda x: abs((x[0] / x[1]) - orig_ratio))
    rw, rh = target_ratio

    # Compute the stride-aligned base size.
    base_h = rh * stride
    base_w = rw * stride
    base_area = base_h * base_w

    # Scale to approximate the target pixel area.
    scale = round(math.sqrt(anchor_pixels / base_area))

    new_h = base_h * scale
    new_w = base_w * scale

    return new_h, new_w


class InternVLUFixResGenerationImageProcessor(BaseImageProcessor):
    r"""
    Image processor that resizes inputs to fixed any-resolution buckets for InternVLU generation.

    Args:
        anchor_pixels (`int`, *optional*, defaults to `512 * 512`):
            Target pixel budget used to pick the nearest any-res bucket.
        anyres_ratio (`str`, *optional*, defaults to `"any_1ratio"`):
            Aspect ratio preset used to select the target bucket.
        stride (`int`, *optional*, defaults to `32`):
            Minimum spatial alignment stride for the target size.
        patch_size (`int`, *optional*, defaults to `8`):
            Patch size used to compute `image_grid_thw`.
    """

    def __init__(
        self,
        anchor_pixels=512 * 512,
        anyres_ratio="any_1ratio",
        stride=32,
        patch_size=8,
    ):
        super().__init__()
        self.anchor_pixels = anchor_pixels
        self.anyres_ratio = anyres_ratio
        self.stride = stride
        self.patch_size = patch_size
        self.normalize = T.Normalize([0.5], [0.5])

    def preprocess(self, images, return_tensors="pt") -> BatchFeature:
        """
        Preprocess images into normalized tensors for fixed-resolution generation.

        Args:
            images (`ImageInput`):
                Image or batch of images to preprocess.
            return_tensors (`str`, *optional*, defaults to `"pt"`):
                The type of tensors to return.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **pixel_values** -- Tensor of shape `(batch, channels, height, width)`.
            - **image_grid_thw** -- Tensor of shape `(batch, 3)` with `(t, h, w)` grid sizes.
        """
        if not isinstance(images, list):
            images = [images]

        processed = []
        for img in images:
            # PIL → Tensor
            tensor = T.ToTensor()(img)
            c, h, w = tensor.shape
            new_h, new_w = dynamic_resize(
                h, w, self.anyres_ratio, self.anchor_pixels, self.stride
            )

            resized = T.Resize(
                (new_h, new_w), interpolation=T.InterpolationMode.BICUBIC
            )(tensor)
            normalized = self.normalize(resized)
            processed.append(normalized)

        batch = torch.stack(processed)

        h_grid = batch.shape[-2] // self.patch_size
        w_grid = batch.shape[-1] // self.patch_size

        image_grid_thw = torch.tensor(
            [[1, h_grid, w_grid] for _ in range(batch.shape[0])],
            dtype=torch.long,
            device=batch.device,
        )

        return BatchFeature(
            data={"pixel_values": batch, "image_grid_thw": image_grid_thw},
            tensor_type=return_tensors,
        )


class InternVLUDynamicResGenerationImageProcessor(BaseImageProcessor):
    r"""
    Image processor that dynamically resizes inputs for InternVLU generation.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions.
        do_pad (`bool`, *optional*, defaults to `False`):
            Whether to pad the image to multiples of `patch_size * merge_size`.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use when resizing the image.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image.
        image_mean (`float` or `List[float]`, *optional*, defaults to `[0.48145466, 0.4578275, 0.40821073]`):
            Mean to use if normalizing the image. This is a float or list of floats for each channel in the image.
        image_std (`float` or `List[float]`, *optional*, defaults to `[0.26862954, 0.26130258, 0.27577711]`):
            Standard deviation to use if normalizing the image. This is a float or list of floats for each channel
            in the image.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
        min_dynamic_patch (`int`, *optional*, defaults to `4096`):
            Minimum number of spatial patches per image before scaling.
        max_dynamic_patch (`int`, *optional*, defaults to `16384`):
            Maximum number of spatial patches per image before scaling.
        max_length (`int`, *optional*):
            Optional maximum side length constraint passed to the resize heuristic.
        vae_downsample_factor (`int`, *optional*, defaults to `8`):
            Patch size for VAE downsampling.
        temporal_patch_size (`int`, *optional*, defaults to `2`):
            Temporal patch size used to tile single images into a pseudo-video grid.
        gen_down_sample_ratio (`int`, *optional*, defaults to `2`):
            Downsample ratio used to compute the merge size for patch packing.
    """

    model_input_names = [
        "pixel_values",
        "image_grid_thw",
        "pixel_values_videos",
        "video_grid_thw",
    ]

    def __init__(
        self,
        do_resize: bool = True,
        do_pad: bool = False,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        min_dynamic_patch: int = 4096,
        max_dynamic_patch: int = 16384,
        max_length: int = None,
        vae_downsample_factor: int = 8,
        temporal_patch_size: int = 2,
        gen_down_sample_ratio: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.do_pad = do_pad
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD

        self.patch_size = vae_downsample_factor
        self.min_pixels = min_dynamic_patch * (self.patch_size**2)
        self.max_pixels = max_dynamic_patch * (self.patch_size**2)

        self.temporal_patch_size = temporal_patch_size
        self.merge_size = int(1.0 / gen_down_sample_ratio)
        self.size = {"min_pixels": self.min_pixels, "max_pixels": self.max_pixels}
        self.do_convert_rgb = do_convert_rgb
        self.max_length = max_length

    def _preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        do_pad: bool = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        min_pixels: int = None,
        max_pixels: int = None,
        max_length: int = None,
    ):
        """
        Preprocess an image or batch of images for dynamic-resolution generation.

        Args:
            images (`ImageInput`):
                Image or batch of images to preprocess. Expects pixel values ranging from 0 to 255. If pixel values
                range from 0 to 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            do_pad (`bool`, *optional*, defaults to `self.do_pad`):
                Whether to pad the image.
            resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the `PILImageResampling` enums.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Scale factor to use if rescaling the image.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Mean to use if normalizing the image. Can be a float or a list of floats corresponding to the number
                of channels in the image.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Standard deviation to use if normalizing the image. Can be a float or a list of floats corresponding
                to the number of channels in the image.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            data_format (`ChannelDimension`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
            min_pixels (`int`, *optional*, defaults to `self.min_pixels`):
                Minimum pixel budget used by the resize heuristic.
            max_pixels (`int`, *optional*, defaults to `self.max_pixels`):
                Maximum pixel budget used by the resize heuristic.
            max_length (`int`, *optional*, defaults to `self.max_length`):
                Maximum side length used by the resize heuristic.

        Returns:
            `Tuple[np.ndarray, Tuple[int, int, int]]`: A tuple with flattened patches and `(t, h, w)` grid metadata.
        """

        min_pixels = min_pixels if min_pixels is not None else self.min_pixels
        max_pixels = max_pixels if max_pixels is not None else self.max_pixels
        max_length = max_length if max_length is not None else self.max_length

        images = make_list_of_images(images)

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if is_scaled_image(images[0]) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )
        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        assert not (
            do_resize and do_pad
        ), "Only one of `do_resize` and `do_pad` can be set to `True`."

        height, width = get_image_size(images[0], channel_dim=input_data_format)
        resized_height, resized_width = height, width
        processed_images = []
        for image in images:
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=self.patch_size * self.merge_size,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                    max_length=max_length,
                )
                image = resize(
                    image,
                    size=(resized_height, resized_width),
                    resample=resample,
                    input_data_format=input_data_format,
                )
            elif do_pad:
                # 1. resize the image s.t. the total number of pixels is within the range [min_pixels, max_pixels] \
                # while maintaining the aspect ratio
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=1,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                    max_length=max_length,
                )
                image = resize(
                    image,
                    size=(resized_height, resized_width),
                    resample=resample,
                    input_data_format=input_data_format,
                )
                # 2. pad the image to the nearest multiple of patch_size * merge_size
                pad_height = (
                    math.ceil(resized_height / (self.patch_size * self.merge_size))
                    * self.patch_size
                    * self.merge_size
                )
                pad_width = (
                    math.ceil(resized_width / (self.patch_size * self.merge_size))
                    * self.patch_size
                    * self.merge_size
                )
                image = pad(
                    image,
                    padding=(
                        (0, pad_height - resized_height),
                        (0, pad_width - resized_width),
                    ),
                    constant_values=0,
                    input_data_format=input_data_format,
                    data_format=input_data_format,
                )
                resized_height, resized_width = pad_height, pad_width

            if do_rescale:
                image = self.rescale(
                    image, scale=rescale_factor, input_data_format=input_data_format
                )

            if do_normalize:
                image = self.normalize(
                    image=image,
                    mean=image_mean,
                    std=image_std,
                    input_data_format=input_data_format,
                )

            image = to_channel_dimension_format(
                image, data_format, input_channel_dim=input_data_format
            )
            processed_images.append(image)

        patches = np.array(processed_images)
        if data_format == ChannelDimension.LAST:
            patches = patches.transpose(0, 3, 1, 2)
        if patches.shape[0] == 1:
            patches = np.tile(patches, (self.temporal_patch_size, 1, 1, 1))
        channel = patches.shape[1]
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h, grid_w = (
            resized_height // self.patch_size,
            resized_width // self.patch_size,
        )
        patches = patches.reshape(
            grid_t,
            self.temporal_patch_size,
            channel,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        )
        # grid_t, grid_h // self.merge_size, grid_w // self.merge_size, \
        # self.merge_size, self.merge_size, channel, self.temporal_patch_size, self.patch_size, self.patch_size
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w,
            channel * self.temporal_patch_size * self.patch_size * self.patch_size,
        )

        return flatten_patches, (grid_t, grid_h, grid_w)

    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        do_pad: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        min_pixels: int = None,
        max_pixels: int = None,
        max_length: int = None,
    ):
        """
        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            do_pad (`bool`, *optional*, defaults to `self.do_pad`):
                Whether to pad the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size metadata kept for validation and API compatibility.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
                has an effect if `do_resize` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
                `True`.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
            min_pixels (`int`, *optional*, defaults to `self.min_pixels`):
                Minimum pixel budget used by the resize heuristic.
            max_pixels (`int`, *optional*, defaults to `self.max_pixels`):
                Maximum pixel budget used by the resize heuristic.
            max_length (`int`, *optional*, defaults to `self.max_length`):
                Maximum side length used by the resize heuristic.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **pixel_values** -- Flattened patch tokens for each image.
            - **image_grid_thw** -- Grid metadata for each image.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        do_pad = do_pad if do_pad is not None else self.do_pad
        size = size if size is not None else self.size
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = (
            rescale_factor if rescale_factor is not None else self.rescale_factor
        )
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = (
            do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        )
        min_pixels = min_pixels if min_pixels is not None else self.min_pixels
        max_pixels = max_pixels if max_pixels is not None else self.max_pixels
        max_length = max_length if max_length is not None else self.max_length

        if images is not None:
            images = make_batched_images(images)

        if images is not None and not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        validate_preprocess_arguments(
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            # do_pad=do_pad,
            size=size,
            resample=resample,
        )

        if images is not None:
            pixel_values, vision_grid_thws = [], []
            for image in images:
                patches, image_grid_thw = self._preprocess(
                    image,
                    do_resize=do_resize,
                    resample=resample,
                    do_rescale=do_rescale,
                    do_pad=do_pad,
                    rescale_factor=rescale_factor,
                    do_normalize=do_normalize,
                    image_mean=image_mean,
                    image_std=image_std,
                    data_format=data_format,
                    do_convert_rgb=do_convert_rgb,
                    input_data_format=input_data_format,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                    max_length=max_length,
                )
                pixel_values.extend(patches)
                vision_grid_thws.append(image_grid_thw)
            pixel_values = np.array(pixel_values)
            vision_grid_thws = np.array(vision_grid_thws)
            data = {"pixel_values": pixel_values, "image_grid_thw": vision_grid_thws}
        return BatchFeature(data=data, tensor_type=return_tensors)


class InternVLUProcessor(ProcessorMixin):
    attributes = [
        "image_processor",
        "image_gen_processor",
        "tokenizer",
        "video_processor",
    ]
    valid_kwargs = [
        "chat_template",
        "image_seq_length",
    ]
    image_processor_class = "AutoImageProcessor"
    image_gen_processor_class = "AutoImageProcessor"
    video_processor_class = "AutoVideoProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor: GotOcr2ImageProcessorFast = None,
        image_gen_processor: Union[
            InternVLUFixResGenerationImageProcessor
            | InternVLUDynamicResGenerationImageProcessor
        ] = None,
        tokenizer: AutoTokenizer = None,
        video_processor: InternVLVideoProcessor = None,
        image_seq_length: int = 256,
        chat_template: str = None,
        blank_messages: dict = {
            "text_blank_message": "Generate an image based on reference images.",
            "pure_blank_message": f"Here is a random image <img_uncond>:",
        },
        **kwargs,
    ):
        self.image_seq_length = image_seq_length
        self.start_image_token = IMG_START_TOKEN
        self.end_image_token = IMG_END_TOKEN
        self.image_token = IMG_CONTEXT_TOKEN
        self.image_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.blank_messages = blank_messages  # dict
        self.image_gen_processor = image_gen_processor

        super().__init__(
            image_processor,
            image_gen_processor,
            tokenizer,
            video_processor,
            chat_template=chat_template,
            **kwargs,
        )

        self.template_name = "qwen2_5-chat-v3"

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # 1. load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
        )

        # 2. load processor_config.json
        config = cls.get_processor_dict(pretrained_model_name_or_path)[0]

        # 3. build image_processor
        image_processor = GotOcr2ImageProcessorFast(**config["image_processor_kwargs"])
        if (
            config["image_gen_processor_class"]
            == "InternVLUDynamicResGenerationImageProcessor"
        ):
            image_gen_processor = InternVLUDynamicResGenerationImageProcessor(
                **config["image_gen_processor_kwargs"]
            )
        elif (
            config["image_gen_processor_class"]
            == "InternVLUFixResGenerationImageProcessor"
        ):
            image_gen_processor = InternVLUFixResGenerationImageProcessor(
                **config["image_gen_processor_kwargs"]
            )
        else:
            raise ValueError(
                f"Unknown image_gen_processor_class {config['image_gen_processor_class']}"
            )

        video_processor = InternVLVideoProcessor()

        return cls(
            tokenizer=tokenizer,
            image_processor=image_processor,
            image_gen_processor=image_gen_processor,
            video_processor=video_processor,
            image_seq_length=config["image_seq_length"],
            chat_template=tokenizer.chat_template,
        )

    @property
    def fix_resolution(self):
        return isinstance(
            self.image_gen_processor, InternVLUFixResGenerationImageProcessor
        )

    def _build_messages(self, text, no_image: bool = False):
        text = text.strip()
        if len(text) != 0:
            text = text[0].upper() + text[1:]
            if text[-1] not in [".", "!", "?", ",", ":", ";"]:
                text = text + "."
        if no_image:
            messages = [
                {
                    "role": "user",
                    "content": text,
                },
            ]
        else:
            if "<image>" not in text:
                text = "<image>\n" + text
            messages = [
                {
                    "role": "user",
                    "content": text,
                },
            ]
        return messages

    def get_drop_messages(
        self, messages, drop_cond_type: DropCondType = DropCondType.none
    ):
        if drop_cond_type == DropCondType.none:
            return messages
        elif drop_cond_type == DropCondType.text:
            contents = messages[-1]["content"]
            if contents.count("<image>") > 0:  # has image
                messages[-1]["content"] = (
                    "<image>\n" * contents.count("<image>")
                    + self.blank_messages["text_blank_message"]
                )
            else:  # for pure text inputs
                messages[-1]["content"] = self.blank_messages["pure_blank_message"]
            return messages

        elif drop_cond_type == DropCondType.all:
            messages[-1]["content"] = self.blank_messages["pure_blank_message"]
            return messages
        else:
            raise NotImplementedError

    def apply_chat_template_with_conv(
        self,
        messages: list,
        conv_template_name: str,
        add_generation_prompt: str = True,
        system_prompt: str = None,
    ):
        conv = get_conv_template(conv_template_name).copy()
        if system_prompt is not None:
            conv.set_system_message(system_prompt)
            logger.warning_once(f"System message was forced to: {system_prompt}")

        for msg in messages:
            if msg["role"] == "user":
                conv.append_message(conv.roles[0], msg["content"])
            elif msg["role"] == "assistant":
                conv.append_message(conv.roles[1], msg["content"])

        if add_generation_prompt:
            conv.append_message(conv.roles[1], "")

        return conv.get_prompt()

    def _insert_media_placeholders(
        self,
        text: list[str],
        image_pixel_values,
        image_num_patches: list[int],
        image_num_patches_indices: np.ndarray,
    ):
        """
        Processes interleaved text with <image> placeholders, replacing them with appropriate
        image tokens while keeping track of the patches used.
        """
        image_index = 0
        processed_text = []
        image_video_patches = []
        replace_strings = []
        for prompt in text:
            new_prompt = prompt
            while self.image_token in new_prompt:
                if self.image_token in new_prompt:
                    # Get the slice of patches corresponding to the current image
                    start_index = (
                        image_num_patches_indices[image_index - 1]
                        if image_index > 0
                        else 0
                    )
                    end_index = image_num_patches_indices[image_index]
                    image_video_patches.append(
                        image_pixel_values[start_index:end_index]
                    )
                    # Replace the corresponding image placeholder with the correct number of image tokens
                    new_prompt = new_prompt.replace(
                        self.image_token, "<placeholder>", 1
                    )
                    replace_strings.append(
                        f"{self.start_image_token}"
                        f"{self.image_token * self.image_seq_length * image_num_patches[image_index]}"
                        f"{self.end_image_token}"
                    )
                    image_index += 1
            while "<placeholder>" in new_prompt:
                replace_str = replace_strings.pop(0)
                new_prompt = new_prompt.replace("<placeholder>", replace_str, 1)
            processed_text.append(new_prompt)
        if (len(image_video_patches) == 0) and (image_pixel_values is not None):
            assert (
                len(image_num_patches_indices) == 1
            ), f"only 1 fake picture in pure text input, but get {len(image_num_patches_indices)}"
            image_video_patches.append(image_pixel_values)
        return processed_text, image_video_patches, image_index

    def __call__(
        self,
        image: Optional[ImageInput] = None,
        prompt: Optional[Union[TextInput, List[TextInput]]] = None,
        generation_mode: Literal["text", "image", "text_image"] = "text",
        height: int = None,
        width: int = None,
        system_prompt: str = None,
        **kwargs: Unpack[InternVLGenProcessorKwargs],
    ) -> BatchFeature:
        """
        Prepare text and/or image inputs for InternVLU generation.

        Args:
            image (`ImageInput`, *optional*):
                Image or batch of images to be prepared. Each image can be a PIL image, NumPy array, or PyTorch tensor.
                Both channels-first and channels-last formats are supported.
            prompt (`TextInput`, `List[TextInput]`, *optional*):
                The sequence or batch of sequences to be encoded. Each sequence can be a string. Required for all
                modes.
            generation_mode (`Literal["text", "image", "text_image"]`):
                Controls whether to prepare text-only, image-only, or combined text-image generation inputs.
            height (`int`, *optional*):
                Desired generation height in pixels. If provided with `width`, overrides default generation size.
            width (`int`, *optional*):
                Desired generation width in pixels. If provided with `height`, overrides default generation size.
            system_prompt (`str`, *optional*):
                Optional system prompt injected into the chat template.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

            kwargs (`InternVLGenProcessorKwargs`):
                Additional keyword arguments forwarded to the tokenizer and image processors.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **pixel_values_gen** -- Optional image tensors used for diffusion when `generation_mode != "text"`.
            - **image_grid_thw_gen** -- Optional generation grid metadata when `generation_mode != "text"`.
            - **generation_flags** -- Flags indicating which image tokens should be generated.
        """

        # Validate inputs.
        if prompt is None:
            raise ValueError("You have to specify text.")
        elif isinstance(prompt, str):
            prompt = [prompt]
        else:
            # Validate each text entry.
            for _ in prompt:
                assert isinstance(_, str)

        if image is None:
            image = [None for _ in range(len(prompt))]
            no_image = True
        elif not isinstance(image, list):
            image = [image]
            no_image = False
        else:
            no_image = False
        assert len(prompt) == len(
            image
        ), f"Recieve {len(prompt)} texts with {len(image)} images"

        if (height is not None) and (width is not None):
            generation_resolution = (height, width)
        elif height is not None:
            generation_resolution = (height, height)
        elif width is not None:
            generation_resolution = (width, width)
        else:
            generation_resolution = DEFAULT_GEN_RESOLUTION  # Default generation size.

        messages_list = []
        for img, t in zip(image, prompt):
            img_list = make_flat_list_of_images([img]) if img is not None else []
            n_imgs = len(img_list)

            if n_imgs > 0:
                n_slots = t.count("<image>")
                if n_slots == 0:
                    t = "<image>\n" * n_imgs + t
                elif n_slots < n_imgs:
                    t = "<image>\n" * (n_imgs - n_slots) + t

            messages = self._build_messages(t, no_image=(n_imgs == 0))
            messages_list.append(messages)

        if generation_mode == "text":
            construct_texts = {
                DropCondType.none.name: [],
            }
            for messages in messages_list:
                temp_prompt = self.apply_chat_template_with_conv(
                    messages,
                    conv_template_name=self.template_name,
                    add_generation_prompt=True,
                    system_prompt=system_prompt,
                )
                construct_texts[DropCondType.none.name].append(
                    temp_prompt.replace("<image>", "<IMG_CONTEXT>")
                )
        else:
            construct_texts = {
                DropCondType.none.name: [],
                DropCondType.text.name: [],
                DropCondType.all.name: [],
            }
            for messages in messages_list:
                for (
                    drop_cond_type_name,
                    drop_cond_type,
                ) in DropCondType.__members__.items():
                    drop_msgs = self.get_drop_messages(messages, drop_cond_type)
                    if no_image:  # t2i task
                        conv_template_name = self.template_name + "-imgen"
                    else:  # editing task
                        conv_template_name = self.template_name + "-editing"
                    temp_prompt = self.apply_chat_template_with_conv(
                        drop_msgs,
                        conv_template_name=conv_template_name,
                        add_generation_prompt=True,
                        system_prompt=system_prompt,
                    )
                    temp_prompt = temp_prompt.replace("<image>", "<IMG_CONTEXT>")
                    if generation_mode == "image":
                        temp_prompt = (
                            temp_prompt + "<img>"
                        )  # text_image task append '<img>' token after cot
                    construct_texts[drop_cond_type_name].append(temp_prompt)
        generation_flags = []

        output_kwargs = self._merge_kwargs(
            InternVLGenProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        image_data_list = []
        gen_data = []
        text_all = []
        for drop_cond_type_name, ctexts in construct_texts.items():
            for text_gen, img in zip(ctexts, image):
                if img is not None:
                    image_num_patches = []
                    image_pixel_values = None
                    image_num_patches_indices = np.array([0])
                    img = make_flat_list_of_images([img])
                    image_inputs = self.image_processor(
                        images=img, **output_kwargs["images_kwargs"]
                    )
                    image_num_patches = image_inputs.pop("num_patches")
                    image_pixel_values = image_inputs.pop("pixel_values")

                    if generation_mode != "text":
                        image_gen_inputs = self.image_gen_processor(
                            images=img, return_tensors="pt"
                        )
                        image_gen_pixel_values = image_gen_inputs.pop("pixel_values")
                        if not self.fix_resolution:
                            image_gen_pixel_values = image_gen_pixel_values.reshape(
                                image_gen_inputs["image_grid_thw"].prod(),
                                -1,
                                self.image_gen_processor.patch_size,
                                self.image_gen_processor.patch_size,
                            )
                        image_grid_thw = image_gen_inputs.pop("image_grid_thw")
                    else:
                        image_gen_pixel_values = None
                        image_grid_thw = None

                    image_num_patches_indices = np.cumsum(image_num_patches)

                    image_data = {}
                    text_gen, image_video_patches, image_index = (
                        self._insert_media_placeholders(
                            [text_gen],
                            image_pixel_values,
                            image_num_patches,
                            image_num_patches_indices,
                        )
                    )
                    text_all.extend(text_gen)
                    if drop_cond_type_name in [
                        DropCondType.none.name,
                        DropCondType.text.name,
                    ]:
                        image_data["pixel_values"] = concatenate_list(
                            image_video_patches
                        )
                        if generation_mode != "text":
                            image_data["pixel_values_gen"] = image_gen_pixel_values
                            image_data["image_grid_thw"] = image_grid_thw
                            gen_data.append(image_grid_thw)

                        image_data_list.append(image_data)
                        generation_flags.extend(
                            [0] * image_data["pixel_values"].shape[0]
                        )

                else:
                    text_all.append(text_gen)

                gen_data.append(
                    torch.tensor(
                        [
                            [
                                1,
                                generation_resolution[0]
                                // self.image_gen_processor.patch_size,
                                generation_resolution[1]
                                // self.image_gen_processor.patch_size,
                            ]
                        ],
                        dtype=torch.long,
                    )
                )
                generation_flags.extend([1])
        # concate data
        batch_data = {}
        if len(image_data_list) > 0:
            batch_data["pixel_values"] = concatenate_list(
                [data["pixel_values"] for data in image_data_list]
            ).to(torch.bfloat16)
            if generation_mode != "text":
                image_grid_thw = concatenate_list(
                    [data["image_grid_thw"] for data in image_data_list]
                )
                pixel_values_gen = concatenate_list(
                    [data["pixel_values_gen"] for data in image_data_list]
                )
                if not self.fix_resolution:
                    pixel_values_gen = pad_images(
                        pixel_values_gen,
                        image_grid_thw,
                        merge_size=self.image_gen_processor.merge_size,
                    ).to(torch.bfloat16)
                batch_data["pixel_values_gen"] = pixel_values_gen
        else:
            image_grid_thw = None

        if generation_mode != "text":
            batch_data["image_grid_thw_gen"] = concatenate_list(gen_data)

        text_inputs = self.tokenizer(
            text_all, **output_kwargs["text_kwargs"]
        )  # "input_ids", "attention_mask"
        batch_data.update({**text_inputs})

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)

        if generation_mode != "text":
            batch_data["generation_flags"] = torch.tensor(
                generation_flags, dtype=torch.long
            )

        return BatchFeature(data=batch_data, tensor_type=return_tensors)
