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

"""Processes an image sequence to a nerfstudio compatible dataset."""

import json
import os
import re
import shutil
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from nerfstudio.process_data import colmap_utils, equirect_utils, process_data_utils
from nerfstudio.process_data.colmap_converter_to_nerfstudio_dataset import (
    ColmapConverterToNerfstudioDataset,
)
from nerfstudio.utils.io import load_from_json
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class LanternImagesToNerfstudioDataset(ColmapConverterToNerfstudioDataset):
    """Process images into a nerfstudio dataset.

    1. Scales images to a specified size.
    2. Calculates the camera poses for each image using `COLMAP <https://colmap.github.io/>`_.
    """

    percent_radius_crop: float = 1.0	
    """Create circle crop mask. The radius is the percent of the image diagonal."""
    # left_colmap_baseline_dir: Optional[Path] = None
    # """Path to the left camera baseline directory."""
    # left_colmap_baseline_mask_dir: Optional[Path] = None
    # """Path to the left camera baseline mask directory."""
    # right_colmap_baseline_dir: Optional[Path] = None
    # """Path to the right camera baseline directory."""
    # right_colmap_baseline_mask_dir: Optional[Path] = None
    # """Path to the right camera baseline mask directory."""
# 
    # left_e1_dir: Optional[Path] = None
    # """Path to the left camera e1 directory (well-exposed)."""
    # left_e1_mask_dir: Optional[Path] = None
    # """Path to the left camera e1 mask directory (well-exposed)."""
    # right_e2_dir: Optional[Path] = None
    # """Path to the right camera e2 directory (fast-exposed)."""
    # right_e2_mask_dir: Optional[Path] = None
    # """Path to the right camera e2 mask directory (fast-exposed)."""

    exposure1: float = 1.0
    """Well-exposed exposure value (left camera)."""

    exposure2: float = 0.009
    """Fast-exposed exposure value (right camera)."""

    def main(self) -> None:
        """Process images into a nerfstudio dataset."""
        assert self.skip_image_processing, "Image processing not supported for now"	
        self.is_HDR = True
        
        require_cameras_exist = False
        if self.colmap_model_path != ColmapConverterToNerfstudioDataset.default_colmap_path():
            if not self.skip_colmap:
                raise RuntimeError("The --colmap-model-path can only be used when --skip-colmap is not set.")
            if not (self.output_dir / self.colmap_model_path).exists():
                raise RuntimeError(f"The colmap-model-path {self.output_dir / self.colmap_model_path} does not exist.")
            require_cameras_exist = True

        image_rename_map: Optional[dict[str, str]] = None

        sequence_names = ['left_colmap_baseline', 'right_colmap_baseline', 'left_e1', 'right_e2']
        sequence_mask_names = ['left_colmap_baseline_mask', 'right_colmap_baseline_mask', 'left_e1_mask', 'right_e2_mask']
        perspective_dirs = []
        perspective_mask_dirs = []

        # Generate planar projections if equirectangular
        if self.camera_type == "equirectangular":
            if self.eval_data is not None:	
                raise ValueError("Cannot use eval_data with camera_type equirectangular.")
            pers_size = equirect_utils.compute_resolution_from_equirect(self.data / sequence_names[0], self.images_per_equirect)
            CONSOLE.log(f"Generating {self.images_per_equirect} {pers_size} sized images per equirectangular image")

            for sequence_name, sequence_mask_name in zip(sequence_names, sequence_mask_names):
                # check if mask exists
                if not (self.data / sequence_mask_name).exists():
                    mask_path = None
                else:
                    mask_path = self.data / sequence_mask_name
                output_dir, output_mask_dir, _ = equirect_utils.generate_planar_projections_from_equirectangular(
                    self.data / sequence_name, pers_size, self.images_per_equirect, mask_path, crop_factor=self.crop_factor
                )
                perspective_dirs.append(output_dir)
                CONSOLE.log(f"Generated {len(os.listdir(output_dir))} images in {output_dir}")
                perspective_mask_dirs.append(output_mask_dir)
                
            self.camera_type = "perspective"
        else:
            raise ValueError("Camera type must be equirectangular")

        summary_log = []

        # Copy and downscale images
        if not self.skip_image_processing:
            # Copy images to output directory
            #TODO 
            image_rename_map_paths = {}
            for i in range(len(perspective_dirs)):
                image_rename_map_paths.update(process_data_utils.copy_images(
                    perspective_dirs[i],
                    image_dir=self.image_dir,
                    crop_factor=self.crop_factor,
                    verbose=self.verbose,
                    num_downscales=self.num_downscales,
                ))
                
                if self.eval_data is not None:
                    raise NotImplementedError
                    eval_image_rename_map_paths = process_data_utils.copy_images(	
                        self.eval_data,	
                        image_dir=self.image_dir,	
                        crop_factor=self.crop_factor,	
                        image_prefix="frame_eval_",	
                        verbose=self.verbose,	
                        num_downscales=self.num_downscales,	
                    )	
                    image_rename_map_paths.update(eval_image_rename_map_paths)
            
            image_rename_map = dict((a.name, b.name) for a, b in image_rename_map_paths.items())
            num_frames = len(image_rename_map)
            summary_log.append(f"Starting with {num_frames} images")

        else:
            num_frames = len(process_data_utils.list_images(self.data / sequence_names[0]))
            if num_frames == 0:
                raise RuntimeError("No usable images in the data folder.")
            summary_log.append(f"Starting with {num_frames} images")

        image_rename_map = None

        with tempfile.TemporaryDirectory() as tmp_dir:
            # copy files for colmap in tmp_dir
            for i in range(len(perspective_dirs)):
                # avoid right_e2 (fast-exposed), colmap doesn't work well on it
                if i == 3:
                    continue

                shutil.copytree(perspective_dirs[i], os.path.join(tmp_dir, sequence_names[i], 'planar_projections'))
            # TODO: mask

            # Run COLMAP
            if not self.skip_colmap:
                require_cameras_exist = True
                self.absolute_colmap_path.mkdir(parents=True, exist_ok=True)
                colmap_utils.run_colmap(
                    image_dir=Path(tmp_dir),
                    colmap_dir=self.absolute_colmap_path,
                    camera_model=process_data_utils.CAMERA_MODELS[self.camera_type],
                    camera_mask_path=None, #TODO
                    gpu=self.gpu,
                    verbose=self.verbose,
                    matching_method=self.matching_method,
                    colmap_cmd=self.colmap_cmd,
                )
                # Colmap uses renamed images
                image_rename_map = None

        # # Export depth maps
        image_id_to_depth_path, log_tmp = self._export_depth()
        summary_log += log_tmp
        incomplete_transforms_file = self.output_dir / "transforms_incomplete.json"

        if not self.skip_colmap:
            if require_cameras_exist and not (self.absolute_colmap_model_path / "cameras.bin").exists():
                raise RuntimeError(f"Could not find existing COLMAP results ({self.colmap_model_path / 'cameras.bin'}).")
            
            summary_log += self._save_transforms(
                num_frames,
                # todo: mask
            )

            shutil.move(str(self.output_dir / 'transforms.json'), incomplete_transforms_file)
            CONSOLE.log(f"Saved initial transforms to {incomplete_transforms_file}")
        original_transforms = load_from_json(incomplete_transforms_file)

        frames = original_transforms["frames"]
        sorted_frames = sorted(frames, key=lambda x: x["file_path"])

        # frames are splitted into 4 groups: left_calib, right_calib, left_e1, left_e2
        # the two first groups are used to infer the basis_change between left and right
        # that basis_change is then applied to the third group (left_e1) to infer the fourth (left_e2)

        grouped_frames = {sequence_name: dict() for sequence_name in ['left_colmap_baseline', 'right_colmap_baseline', 'left_e1']}
        for frame in sorted_frames:
            # use regex
            matches = re.match(r'images/(\w+)/.*_(\d+)_(\d+).png', frame["file_path"])
            if matches:
                sequence_name = matches.group(1)
                frame_idx = int(matches.group(2))
                crop_direction = int(matches.group(3))
                if crop_direction not in grouped_frames[sequence_name].keys():
                    grouped_frames[sequence_name][crop_direction] = []
                grouped_frames[sequence_name][crop_direction].append(frame)
            else:
                # warning
                print(f'Warning: {frame["file_path"]} does not match regex')


        # group 3 is empty, we want to infer its transform
        grouped_frames['right_e2'] = dict()
        for crop_direction in range(self.images_per_equirect):
            group_0_tranform = np.array(grouped_frames['left_colmap_baseline'][crop_direction][0]['transform_matrix'])
            group_1_tranform = np.array(grouped_frames['right_colmap_baseline'][crop_direction][0]['transform_matrix'])
            
            # TODO: check which one is right
            basis_change = np.linalg.inv(group_0_tranform) @ group_1_tranform
            # basis_change = np.linalg.inv(group_1_tranform) @ group_0_tranform

            # sometimes, colmap omits frames, so if a frame doesn't have the left_e1 pose, we can't infer the right_e2 pose
            if grouped_frames['left_e1'].get(crop_direction):
                for frame in grouped_frames['left_e1'][crop_direction]:
                    
                    # apply group_3_tranform to frame, put it in group 3
                    group_3_frame_transform = (np.array(frame['transform_matrix']) @ basis_change).tolist()
                    if crop_direction not in grouped_frames['right_e2'].keys():
                        grouped_frames['right_e2'][crop_direction] = []
                    grouped_frames['right_e2'][crop_direction].append({
                        'file_path': frame['file_path'].replace('left_e1', 'right_e2').replace('lhs', 'rhs'), # TODO validate file naming convention
                        'transform_matrix': group_3_frame_transform
                    })
            else:
                # warning
                print(f'Warning: no left_e1 for crop_direction {crop_direction}')
        
        # write to text file grouped_frames
        with open(self.output_dir / 'grouped_frames.json', 'w') as f:
            json.dump(grouped_frames, f, indent=4)

        # generate new transforms.json
        new_transforms = deepcopy(original_transforms)
        new_transforms['frames'] = []
        for sequence_name in ['left_e1', 'right_e2']:
            for i, crop_direction in enumerate(range(self.images_per_equirect)):
                new_transforms['frames'] += grouped_frames[sequence_name][crop_direction]
                
        with open(self.output_dir / 'transforms.json', "w", encoding="UTF-8") as file:
            json.dump(new_transforms, file, indent=4)

        # for HDR-Nerfacto, output also the exposures
        exposures_content = {}
        for frame in new_transforms['frames']:
            if 'right_e2' in frame['file_path']:
                exposures_content[frame['file_path']] = self.exposure2
            else:
                exposures_content[frame['file_path']] = self.exposure1
        
        with open(self.output_dir / 'exposures.json', "w", encoding="UTF-8") as file:
            json.dump(exposures_content, file, indent=4)        

        CONSOLE.log("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")

        for summary in summary_log:
            CONSOLE.log(summary)
