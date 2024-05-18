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
Code for sampling pixels.
"""

import random
from dataclasses import dataclass, field
from math import ceil
from typing import (
    Dict,
    Optional,
    Type,
    Union,
)

import torch
from jaxtyping import Int
from torch import Tensor

from nerfstudio.configs.base_config import (
    InstantiateConfig,
)
from nerfstudio.data.utils.pixel_sampling_utils import erode_mask
from nerfstudio.utils import profiler


@dataclass
class PixelSamplerConfig(InstantiateConfig):
    """Configuration for pixel sampler instantiation."""

    _target: Type = field(default_factory=lambda: PixelSampler)
    """Target class to instantiate."""
    num_rays_per_batch: int = 4096
    """Number of rays to sample per batch."""
    keep_full_image: bool = False
    """Whether or not to include a reference to the full image in returned batch."""
    is_equirectangular: bool = False
    """List of whether or not camera i is equirectangular."""

    lantern_steps: int = 1
    """  Define which steps of lantern is going on!! """


class PixelSampler:
    """Samples 'pixel_batch's from 'image_batch's.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: PixelSamplerConfig
    is_well_exposed_computed: bool = False
    is_fast_exposed_computed: bool = False
    is_light_sources_computed: bool = False
    well_exposed_indices: torch.Tensor
    fast_exposed_indices: torch.Tensor
    light_sources_indices: torch.Tensor
    fast_image_batch = {}
    well_image_batch = {}
    def __init__(self, config: PixelSamplerConfig, **kwargs) -> None:
        self.kwargs = kwargs
        self.config = config
        # Possibly override some values if they are present in the kwargs dictionary
        self.config.num_rays_per_batch = self.kwargs.get("num_rays_per_batch", self.config.num_rays_per_batch)
        self.config.keep_full_image = self.kwargs.get("keep_full_image", self.config.keep_full_image)
        self.config.is_equirectangular = self.kwargs.get("is_equirectangular", self.config.is_equirectangular)
        self.set_num_rays_per_batch(self.config.num_rays_per_batch)

    def set_num_rays_per_batch(self, num_rays_per_batch: int):
        """Set the number of rays to sample per batch.

        Args:
            num_rays_per_batch: number of rays to sample per batch
        """
        self.num_rays_per_batch = num_rays_per_batch


    def sample_considering_mask(self, mask, device, batch_size):
        num_masks = mask.shape[0]
        indices = torch.tensor([]).cuda().to(device)
        nonzero_indices_whole = torch.tensor([]).to(device)

        if num_masks > 0:
            size_of_disk = 50
            num_disks = ceil(num_masks / size_of_disk)
            # slice mask to 100 disk
            random_idx_disk = int(torch.randint(0, num_disks, (1,)))
            
            for current_disk, start in enumerate(range(0, num_masks, size_of_disk)):
                is_last_disk = ((start//size_of_disk) == num_disks-1)
                end = start + size_of_disk if not is_last_disk else num_masks
                current_masks = mask[start:end,...].to(device)
                nonzero_indices = torch.nonzero(current_masks, as_tuple=False)
                # Add offset to indices:local indices --> global
                nonzero_indices[:,0] += start
                
                num_samples = batch_size // (num_disks-1) if current_disk != random_idx_disk \
                                    else batch_size % (num_disks-1)
                chosen_indices = torch.randint(0, nonzero_indices.shape[0], (num_samples,), device=device)
                
                indice_this_disk = nonzero_indices[chosen_indices]
                indices = torch.cat((indices, indice_this_disk), dim=0).long()
                nonzero_indices_whole = torch.cat((nonzero_indices_whole, nonzero_indices.cpu()), dim=0)
                
                del current_masks
                del nonzero_indices
                del indice_this_disk
        
        return indices, nonzero_indices_whole
        
        
    @profiler.time_function
    def sample_method(
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
        is_fast_expo: bool = False,
        only_light_sources: bool = False,
        validity: Optional[Tensor] = None,
    ) -> Int[Tensor, "batch_size 3"]:
        """
        Naive pixel sampler, uniformly samples across all possible pixels of all possible images.

        Args:
            batch_size: number of samples in a batch
            num_images: number of images to sample over
            mask: mask of possible pixels in an image to sample from.
        """
        # TODO Defining a flag for sampling even when there is a mask 
        if isinstance(mask, torch.Tensor):
            # mask size: batch * h * w * 1
            # Init state, cache masks indices.

            # if self.states_of_mask != 2:    
            with profiler.time_function("mask squeeze: whole"):
                mask = mask.squeeze(-1)
                
            if is_fast_expo:
                if only_light_sources:
                    if self.is_light_sources_computed:
                        chosen_indices = torch.randint(0, self.light_sources_indices.shape[0], (batch_size,))
                        return self.light_sources_indices[chosen_indices].to(device)
                    else:
                        assert validity is not None, "Validy needs to be defined for the light sources samples"
                        validity = validity.squeeze(-1).cpu()
                        mask_combine = torch.logical_and(mask, validity)
                        indices, nonzero_indices_whole = self.sample_considering_mask(mask_combine, device, batch_size)
                        self.light_sources_indices = nonzero_indices_whole.long().cpu()
                        self.is_light_sources_computed = True
                        return indices
                else:
                    if self.is_fast_exposed_computed:
                        chosen_indices = torch.randint(0, self.fast_exposed_indices.shape[0], (batch_size,))
                        return self.fast_exposed_indices[chosen_indices].to(device)
                    else:
                        indices, nonzero_indices_whole = self.sample_considering_mask(mask, device, batch_size)
                        self.fast_exposed_indices = nonzero_indices_whole.long().cpu()
                        self.is_fast_exposed_computed = True
                        return indices
            else:
                if self.is_well_exposed_computed:
                    chosen_indices = torch.randint(0, self.well_exposed_indices.shape[0], (batch_size,))
                    return self.well_exposed_indices[chosen_indices].to(device)
                else:
                    indices, nonzero_indices_whole = self.sample_considering_mask(mask, device, batch_size)
                    self.well_exposed_indices = nonzero_indices_whole.long().cpu()
                    self.is_well_exposed_computed = True
                    return indices
        else:
            indices = torch.floor(
                torch.rand((batch_size, 3), device=device)
                * torch.tensor([num_images, image_height, image_width], device=device)
            ).long()

        return indices

    def sample_method_equirectangular(
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:
        if isinstance(mask, torch.Tensor):
            # Note: if there is a mask, sampling reduces back to uniform sampling, which gives more
            # sampling weight to the poles of the image than the equators.
            # TODO(kevinddchen): implement the correct mask-sampling method.

            indices = self.sample_method(batch_size, num_images, image_height, image_width, mask=mask, device=device)
        else:
            # We sample theta uniformly in [0, 2*pi]
            # We sample phi in [0, pi] according to the PDF f(phi) = sin(phi) / 2.
            # This is done by inverse transform sampling.
            # http://corysimon.github.io/articles/uniformdistn-on-sphere/
            num_images_rand = torch.rand(batch_size, device=device)
            phi_rand = torch.acos(1 - 2 * torch.rand(batch_size, device=device)) / torch.pi
            theta_rand = torch.rand(batch_size, device=device)
            indices = torch.floor(
                torch.stack((num_images_rand, phi_rand, theta_rand), dim=-1)
                * torch.tensor([num_images, image_height, image_width], device=device)
            ).long()

        return indices

    def collate_image_dataset_batch(self, batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False, is_fast_expo: bool = False, only_light_sources: bool = False): 
        """
        Operates on a batch of images and samples pixels to use for generating rays.
        Returns a collated batch which is input to the Graph.
        It will sample only within the valid 'mask' if it's specified.

        Args:
            batch: batch of images to sample from
            num_rays_per_batch: number of rays to sample per batch
            keep_full_image: whether or not to include a reference to the full image in returned batch
        """

        device = batch["image"].device
        num_images, image_height, image_width, _ = batch["image"].shape

        if "mask" in batch:
            if self.config.is_equirectangular:
                indices = self.sample_method_equirectangular(
                    num_rays_per_batch, num_images, image_height, image_width, mask=batch["mask"], device=device
                )
            else:
                if "saturation_mask" in batch:
                    indices = self.sample_method(
                        num_rays_per_batch, num_images, image_height, image_width, mask=batch["mask"], device=device, 
                        is_fast_expo=is_fast_expo, 
                        only_light_sources=only_light_sources,
                        validity=batch["saturation_mask"]
                    )
                else:
                    indices = self.sample_method(
                        num_rays_per_batch, num_images, image_height, image_width, mask=batch["mask"], device=device, 
                        is_fast_expo=is_fast_expo, 
                        only_light_sources=only_light_sources,
                        validity=None
                    )
        else:
            if self.config.is_equirectangular:
                indices = self.sample_method_equirectangular(
                    num_rays_per_batch, num_images, image_height, image_width, device=device
                )
            else:
                indices = self.sample_method(num_rays_per_batch, num_images, image_height, image_width, device=device)

        chan, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        chan, y, x = chan.cpu(), y.cpu(), x.cpu()
        collated_batch = {
            key: value[chan, y, x] for key, value in batch.items() if key != "image_idx" and key != "exposure" and key != "image_filename" and value is not None
        }

        assert collated_batch["image"].shape[0] == num_rays_per_batch

        # Needed to correct the random indices to their actual camera idx locations.
        collated_batch["indices"] = indices.clone()  # with the abs camera indices
        collated_batch["indices"][:, 0] = batch["image_idx"][chan].clone()

        if "exposure" in batch:
            collated_batch["exposure"] = batch["exposure"][chan]
        
        if keep_full_image:
            collated_batch["full_image"] = batch["image"]

        return collated_batch

    def collate_image_dataset_batch_list(self, batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
        """
        Does the same as collate_image_dataset_batch, except it will operate over a list of images / masks inside
        a list.

        We will use this with the intent of DEPRECIATING it as soon as we find a viable alternative.
        The intention will be to replace this with a more efficient implementation that doesn't require a for loop, but
        since pytorch's ragged tensors are still in beta (this would allow for some vectorization), this will do.

        Args:
            batch: batch of images to sample from
            num_rays_per_batch: number of rays to sample per batch
            keep_full_image: whether or not to include a reference to the full image in returned batch
        """

        device = batch["image"][0].device
        num_images = len(batch["image"])

        # only sample within the mask, if the mask is in the batch
        all_indices = []
        all_images = []

        if "mask" in batch:
            num_rays_in_batch = num_rays_per_batch // num_images
            for i in range(num_images):
                image_height, image_width, _ = batch["image"][i].shape

                if i == num_images - 1:
                    num_rays_in_batch = num_rays_per_batch - (num_images - 1) * num_rays_in_batch

                indices = self.sample_method(
                    num_rays_in_batch, 1, image_height, image_width, mask=batch["mask"][i], device=device
                )
                indices[:, 0] = i
                all_indices.append(indices)
                all_images.append(batch["image"][i][indices[:, 1], indices[:, 2]])

        else:
            num_rays_in_batch = num_rays_per_batch // num_images
            for i in range(num_images):
                image_height, image_width, _ = batch["image"][i].shape
                if i == num_images - 1:
                    num_rays_in_batch = num_rays_per_batch - (num_images - 1) * num_rays_in_batch
                if self.config.is_equirectangular:
                    indices = self.sample_method_equirectangular(
                        num_rays_in_batch, 1, image_height, image_width, device=device
                    )
                else:
                    indices = self.sample_method(num_rays_in_batch, 1, image_height, image_width, device=device)
                indices[:, 0] = i
                all_indices.append(indices)
                all_images.append(batch["image"][i][indices[:, 1], indices[:, 2]])

        indices = torch.cat(all_indices, dim=0)

        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        collated_batch = {
            key: value[c, y, x]
            for key, value in batch.items()
            if key != "image_idx" and key != "image" and key != "mask" and value is not None
        }

        collated_batch["image"] = torch.cat(all_images, dim=0)

        assert collated_batch["image"].shape[0] == num_rays_per_batch

        # Needed to correct the random indices to their actual camera idx locations.
        indices[:, 0] = batch["image_idx"][c]
        collated_batch["indices"] = indices  # with the abs camera indices

        if keep_full_image:
            collated_batch["full_image"] = batch["image"]

        return collated_batch

    def sample(self, image_batch: Dict):
        """Sample an image batch and return a pixel batch.

        Args:
            image_batch: batch of images to sample from
        """
        if isinstance(image_batch["image"], list):
            image_batch = dict(image_batch.items())  # copy the dictionary so we don't modify the original
            pixel_batch = self.collate_image_dataset_batch_list(
                image_batch, self.num_rays_per_batch, keep_full_image=self.config.keep_full_image
            )
        elif isinstance(image_batch["image"], torch.Tensor):
            if "exposure" in image_batch:
                if not(self.well_image_batch and self.fast_image_batch):
                    well_ones = (image_batch["exposure"] == 1.0).cpu()
                    fast_ones = ~well_ones
                    for key in image_batch.keys() - ['image_filename']:
                        self.well_image_batch[key] = image_batch[key][well_ones]
                        self.fast_image_batch[key] = image_batch[key][fast_ones]
                
                assert self.config.lantern_steps == 1 or self.config.lantern_steps == 2, "The step of Lantern should be either 1 or 2!! "
                
                if self.config.lantern_steps == 1:
                    well_pixel_batch = self.collate_image_dataset_batch(
                        self.well_image_batch, int(1.0 * self.num_rays_per_batch), keep_full_image=self.config.keep_full_image
                    )
                    fast_pixel_batch = self.collate_image_dataset_batch(
                        self.fast_image_batch, int(0.0 * self.num_rays_per_batch), keep_full_image=self.config.keep_full_image, is_fast_expo=True
                    )
                    light_pixel_batch = self.collate_image_dataset_batch(
                        self.fast_image_batch, int(0.0 * self.num_rays_per_batch), keep_full_image=self.config.keep_full_image, is_fast_expo=True, only_light_sources=True
                    )
                else:
                    well_pixel_batch = self.collate_image_dataset_batch(
                        self.well_image_batch, int(0.2 * self.num_rays_per_batch), keep_full_image=self.config.keep_full_image
                    )
                    fast_pixel_batch = self.collate_image_dataset_batch(
                        self.fast_image_batch, int(0.3 * self.num_rays_per_batch), keep_full_image=self.config.keep_full_image, is_fast_expo=True
                    )
                    light_pixel_batch = self.collate_image_dataset_batch(
                        self.fast_image_batch, int(0.5 * self.num_rays_per_batch), keep_full_image=self.config.keep_full_image, is_fast_expo=True, only_light_sources=True
                    )
                    # well_pixel_batch = self.collate_image_dataset_batch(
                    #     self.well_image_batch, int(0.0 * self.num_rays_per_batch), keep_full_image=self.config.keep_full_image
                    # )
                    # fast_pixel_batch = self.collate_image_dataset_batch(
                    #     self.fast_image_batch, int(0.3 * self.num_rays_per_batch), keep_full_image=self.config.keep_full_image, is_fast_expo=True
                    # )
                    # light_pixel_batch = self.collate_image_dataset_batch(
                    #     self.fast_image_batch, int(0.7 * self.num_rays_per_batch), keep_full_image=self.config.keep_full_image, is_fast_expo=True, only_light_sources=True
                    # )
                
                pixel_batch = well_pixel_batch #.copy()
                for key, value in fast_pixel_batch.items():
                    pixel_batch[key] = torch.cat((pixel_batch[key], value), 0)
                for key, value in light_pixel_batch.items():
                    pixel_batch[key] = torch.cat((pixel_batch[key], value), 0)
            else:
                pixel_batch = self.collate_image_dataset_batch(
                    image_batch, self.num_rays_per_batch, keep_full_image=self.config.keep_full_image
                )
        else:
            raise ValueError("image_batch['image'] must be a list or torch.Tensor")
        return pixel_batch


@dataclass
class PatchPixelSamplerConfig(PixelSamplerConfig):
    """Config dataclass for PatchPixelSampler."""

    _target: Type = field(default_factory=lambda: PatchPixelSampler)
    """Target class to instantiate."""
    patch_size: int = 32
    """Side length of patch. This must be consistent in the method
    config in order for samples to be reshaped into patches correctly."""


class PatchPixelSampler(PixelSampler):
    """Samples 'pixel_batch's from 'image_batch's. Samples square patches
    from the images randomly. Useful for patch-based losses.

    Args:
        config: the PatchPixelSamplerConfig used to instantiate class
    """

    config: PatchPixelSamplerConfig

    def set_num_rays_per_batch(self, num_rays_per_batch: int):
        """Set the number of rays to sample per batch. Overridden to deal with patch-based sampling.

        Args:
            num_rays_per_batch: number of rays to sample per batch
        """
        self.num_rays_per_batch = (num_rays_per_batch // (self.config.patch_size**2)) * (self.config.patch_size**2)
        
    # overrides base method
    def sample_method(
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:
        if isinstance(mask, Tensor):
            sub_bs = batch_size // (self.config.patch_size**2)
            half_patch_size = int(self.config.patch_size / 2)
            # m = erode_mask(mask.permute(0, 3, 1, 2).float(), pixel_radius=half_patch_size)
            m = mask.permute(0, 3, 1, 2).float()
            nonzero_indices = torch.nonzero(m[:, 0], as_tuple=False).to(device)
            chosen_indices = random.sample(range(len(nonzero_indices)), k=sub_bs)
            indices = nonzero_indices[chosen_indices]

            indices = (
                indices.view(sub_bs, 1, 1, 3)
                .broadcast_to(sub_bs, self.config.patch_size, self.config.patch_size, 3)
                .clone()
            )

            yys, xxs = torch.meshgrid(
                torch.arange(self.config.patch_size, device=device), torch.arange(self.config.patch_size, device=device)
            )
            indices[:, ..., 1] += yys - half_patch_size
            indices[:, ..., 2] += xxs - half_patch_size

            indices = torch.floor(indices).long()
            indices = indices.flatten(0, 2)
        else:
            sub_bs = batch_size // (self.config.patch_size**2)
            indices = torch.rand((sub_bs, 3), device=device) * torch.tensor(
                [num_images, image_height - self.config.patch_size, image_width - self.config.patch_size],
                device=device,
            )

            indices = (
                indices.view(sub_bs, 1, 1, 3)
                .broadcast_to(sub_bs, self.config.patch_size, self.config.patch_size, 3)
                .clone()
            )

            yys, xxs = torch.meshgrid(
                torch.arange(self.config.patch_size, device=device), torch.arange(self.config.patch_size, device=device)
            )
            indices[:, ..., 1] += yys
            indices[:, ..., 2] += xxs

            indices = torch.floor(indices).long()
            indices = indices.flatten(0, 2)

        return indices


@dataclass
class PairPixelSamplerConfig(PixelSamplerConfig):
    """Config dataclass for PairPixelSampler."""

    _target: Type = field(default_factory=lambda: PairPixelSampler)
    """Target class to instantiate."""
    radius: int = 2
    """max distance between pairs of pixels."""


class PairPixelSampler(PixelSampler):  # pylint: disable=too-few-public-methods
    """Samples pair of pixels from 'image_batch's. Samples pairs of pixels from
        from the images randomly within a 'radius' distance apart. Useful for pair-based losses.

    Args:
        config: the PairPixelSamplerConfig used to instantiate class
    """

    def __init__(self, config: PairPixelSamplerConfig, **kwargs) -> None:
        self.config = config
        self.radius = self.config.radius
        super().__init__(self.config, **kwargs)
        self.rays_to_sample = self.config.num_rays_per_batch // 2

    # overrides base method
    def sample_method(  # pylint: disable=no-self-use
        self,
        batch_size: Optional[int],
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:
        if isinstance(mask, Tensor):
            # m = erode_mask(mask.permute(0, 3, 1, 2).float(), pixel_radius=self.radius)
            m = mask.permute(0, 3, 1, 2).float()
            nonzero_indices = torch.nonzero(m[:, 0], as_tuple=False).to(device)
            chosen_indices = random.sample(range(len(nonzero_indices)), k=self.rays_to_sample)
            indices = nonzero_indices[chosen_indices]
        else:
            rays_to_sample = self.rays_to_sample
            if batch_size is not None:
                assert (
                    int(batch_size) % 2 == 0
                ), f"PairPixelSampler can only return batch sizes in multiples of two (got {batch_size})"
                rays_to_sample = batch_size // 2

            s = (rays_to_sample, 1)
            ns = torch.randint(0, num_images, s, dtype=torch.long, device=device)
            hs = torch.randint(self.radius, image_height - self.radius, s, dtype=torch.long, device=device)
            ws = torch.randint(self.radius, image_width - self.radius, s, dtype=torch.long, device=device)
            indices = torch.concat((ns, hs, ws), dim=1)

            pair_indices = torch.hstack(
                (
                    torch.zeros(rays_to_sample, 1, device=device, dtype=torch.long),
                    torch.randint(-self.radius, self.radius, (rays_to_sample, 2), device=device, dtype=torch.long),
                )
            )
            pair_indices += indices
            indices = torch.hstack((indices, pair_indices)).view(rays_to_sample * 2, 3)
        return indices