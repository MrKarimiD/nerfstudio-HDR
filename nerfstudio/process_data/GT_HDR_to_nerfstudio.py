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

"""Processes an panorama sequence to a nerfstudio compatible dataset."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal, Tuple
import os
import numpy as np
from numpy.linalg import inv
import cv2
from tqdm import tqdm


from nerfstudio.process_data import equirect_utils, process_data_utils
from nerfstudio.process_data.base_converter_to_nerfstudio_dataset import (
    BaseConverterToNerfstudioDataset,
)
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from nerfstudio.utils.rich_utils import CONSOLE, ItersPerSecColumn
from nerfstudio.utils import io


def apply_correction(coefs, img, CORRECTION_CURVE_TYPE) -> np.ndarray:
    # check that img is in range 0-255
    assert img.dtype == np.uint8
    assert CORRECTION_CURVE_TYPE in ['gamma']
    img = ((img.astype(np.float32) / 255.0) ** np.array(coefs).T) # * 255.0
    # img = ((img.astype(np.float32) / 255.0) ** coefs[None, None, :]) * 255.0
    return img


def distance(point_array, ref):
    pow_2 = np.power(point_array - ref, 2)
    sum_pow_2 = np.sum(pow_2, axis=-1)
    dist = np.sqrt(sum_pow_2)
    return dist


def opensfm_to_opengl(shot_data):
    rvec = np.array(tuple(map(float, shot_data['rotation'])))
    tvec = np.array(tuple(map(float, shot_data['translation'])))

    # Converting opencv to opengl coordinates
    from scipy.spatial.transform import Rotation as R
    rot = R.from_rotvec(rvec).as_matrix()
    
    W2C = np.eye(4)
    W2C[:3, :3] = rot
    W2C[:3, 3] = np.array(tvec)
    
    C2W = inv(W2C)
    C2W[0:3, 1:3] *= -1
    C2W = C2W[np.array([1, 0, 2, 3]), :]
    C2W[2, :] *= -1

    return C2W, rvec, tvec


@dataclass
class GTHDRoNeRFStudioDataset(BaseConverterToNerfstudioDataset):
    """Process panorama data into a nerfstudio dataset. Skip the COLMAP

    This script does the following:

    1. Crops panoramas to planar images, and restore corresponding perspective camera poses.
    1. Scales images to a specified size.
    2. Converts perpective camera poses into the nerfstudio format.
    """
    
    metadata: Path = ""
    """Path the metadata of the panoramas sequence."""

    checkpoint: Path = ""
    """Path the metadata of the panoramas sequence."""
    
    
    def main(self) -> None:
        """Process images into a nerfstudio dataset."""

        summary_log = []

        openSFM_reconstruction = io.load_from_json(self.metadata)
        frames_names = openSFM_reconstruction[0]['shots'].keys()
        frames_names = sorted(frames_names)
        
        dataset_transform = io.load_from_json(self.checkpoint / "dataparser_transforms.json")
        scale_factor = dataset_transform['scale']
        normalization_mat = np.eye(4)
        normalization_mat[:3, :] = np.array(dataset_transform['transform'])
        
        # Convert OpenSFM coordinates to the NeRFStudio
        camera_dict = {}
        camera_dict["fps"] = 1
        camera_key = 'v2 unknown unknown -1 -1 perspective 0'
        camera_dict["camera_type"] = "equirectangular"
        camera_dict["render_width"] = openSFM_reconstruction[0]["cameras"][camera_key]['width']
        camera_dict["render_height"] = openSFM_reconstruction[0]["cameras"][camera_key]['height']
        camera_dict["smoothness_value"] = 0
        camera_dict["is_cycle"] = False
        camera_dict["camera_path"] = []
        
        frame_length = 0
        for idx, fname in enumerate([x for x in frames_names if x.startswith('GT')]):
            print(fname)
            shot_data = openSFM_reconstruction[0]['shots'][fname]
            C2W, rvec, tvec = opensfm_to_opengl(shot_data)
            C2W = normalization_mat @ C2W 
            C2W[:3, 3] *= scale_factor
            camera_dict["camera_path"].append(
            {
                "camera_to_world": C2W.flatten().tolist(),
                "fov": 20.407948880476418,
                "aspect": 1,
            })
            frame_length += 1            
        camera_dict["seconds"] = float(frame_length)
        
        with open(self.output_dir / 'GT_transforms.json', 'w') as f:
            json.dump(camera_dict, f, indent=4)
        
        CONSOLE.rule("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")

        for summary in summary_log:
            CONSOLE.print(summary, justify="center")
        CONSOLE.rule()