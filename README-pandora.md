

# How it works?

## 0. Setup

```

pip install skylibs equilib
```

## 1. Data acquisition & pre-processing

### OPTION 1: Real data

1. Acquire data using the apparatus (stick with two Ricoh cameras)

    Make sure the cameras point up to avoid having lights in the seam.

    Do two recordings with both Ricoh Theta cameras. Use the flash of a flashlight both times, the falling edge will later be detected by a software.

    > Cameras:
    > 
    > **camera 1**: left, well-exposed, serial: YN14100695 \
    > **camera 3**: right, under-exposed, serial: YN14111000

2. Take one video capture with the two cameras well-exposed.

    Use flash at the beginning (and at the end?) to use for synchronization.

3. Take one video capture with Camera 1 (left) well-exposed and Camera 2 (right) fast-exposed.

    Use flash at the beginning (and at the end?) to use for synchronization.

4. Run the script for automatic flash detection / use Shotcut to manually align and trim the videos manually.
    
    It will temporally align the frames from the left and right camera and remove the initial frames with the flash.

5. Extract frames using ffmpeg

    ```
    python mp4_videos_to_png_frames.py
    ```

    It should output the following file structure
    ```
    left_colmap_baseline/
        ....png
        ....png
        ....png
    right_colmap_baseline/
        ....png
        ....png
        ....png
    left_e1/
        ....png
        ....png
        ....png
    right_right_e1/
        ....png
        ....png
        ....png
    ```
    
6. Run the code to mask out people and the stick. You may need to specify `--input-dir` and `--output-dir`

    ```
    python lantern_scripts/mask_humans2.py
    ```


### OPTION 2: Synthetic data

``
export PYTHONPATH=$pwd:$PYTHONPATH
python lantern_scripts/split_synthetic_colmap.py --has_exposure --left_dir ./Dual_Cameras/left --right_dir ./Dual_Cameras/right --out_dir ./Dual_Cameras/split_colmap_out
``

## 2. Process data using COLMAP

Example with synthetic data:

```
ns-process-data lantern --data ./Dual_Cameras/split_colmap_subset --output-dir ./Dual_Cameras/split_colmap_subset/split_colmap_out5 --skip-image-processing  --camera-type equirectangular --images-per-equirect 8 
```

Example with real data (note that `--mask-dir` was added):

```
ns-process-data lantern --data ./data/lab_ground_floor/trimmed_videos --mask-dir ./data/lab_ground_floor/trimmed_videos_masks --output-dir ./data/lab_ground_floor/colmap_out --skip-image-processing  --camera-type equirectangular --images-per-equirect 8 
```

## 2. Pre-training NeRF

Using both the well-exposed and fast-exposed frames, as well as their associated camera poses, a NeRF will be trained to predict both the RGB and mask to do volumetric rendering.

After convergence, go to the next step.

```
ns-train lantern-nerfacto --help
```

> Thing to keep in mind: the way Nerfstudio splits train and test images may not be optimal: left might be in one dataset and right in another... that's a bit of a data leak.


## 3. Running MomoNet

(Using which frames?) A pretrained UNet image-to-image network called MomoNet will infill the missing dynamic range and store these predictions in new images.
It will be trained on the Laval indoor HDR dataset to do this task.

TODO: validate this part

## 4. Train from scratch or fine-tune the NeRF model using MomoNet's predictions



```
```