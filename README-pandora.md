

# How it works?

## 0. Setup

```
export PYTHONPATH=$pwd:$PYTHONPATH
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
    python lantern_scripts/mp4_videos_to_png_frames.py --help
    python lantern_scripts/mp4_videos_to_png_frames.py --input_dir data/lab_downstairs/trimmed_videos
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
    right_e2/
        ....png
        ....png
        ....png
    ```
    
6. Run the code to mask out people and the stick.

    Make sure the masks in "./stick_masks" are valid
    ```
    python lantern_scripts/mask_humans2.py --help
    python lantern_scripts/mask_humans2.py --input_dir data/lab_downstairs/trimmed_videos
    ```


### OPTION 2: Synthetic data

``
export PYTHONPATH=$pwd:$PYTHONPATH
python lantern_scripts/split_synthetic_colmap.py --has_exposure --left_dir ./Dual_Cameras/left --right_dir ./Dual_Cameras/right --out_dir ./Dual_Cameras/split_colmap_out
``

## 2. Process data using COLMAP

Example with synthetic data:

```
export PYTHONPATH=$pwd:$PYTHONPATH
ns-process-data lantern --data ./Dual_Cameras/split_colmap_subset --output-dir ./Dual_Cameras/split_colmap_subset/split_colmap_out5 --skip-image-processing  --camera-type equirectangular --images-per-equirect 8 
```

Example with real data (note that `--mask-dir` was added):

```
export PYTHONPATH=$pwd:$PYTHONPATH
ns-process-data lantern --data ./data/lab_downstairs/captured_data --mask-dir ./data/lab_downstairs/masks --output-dir ./data/lab_downstairs/colmap_out --skip-image-processing  --camera-type equirectangular --images-per-equirect 8
```

After that, you'll need to copy manually (for now), the folder to `images`, for nerf training.

Examples
```
ns-process-data lantern --data ./data/lab_ground_floor/trimmed_videos_smart_subset  --mask-dir ./data/lab_ground_floor/trimmed_videos_masks --output-dir ./data/lab_ground_floor/colmap_out_smart --skip-image-processing  --camera-type equirectangular --images-per-equirect 8 --skip-colmap --skip-perspective-transform
```

## 2. Pre-training NeRF

Using both the well-exposed and fast-exposed frames, as well as their associated camera poses, a NeRF will be trained to predict both the RGB and mask to do volumetric rendering.

After convergence, go to the next step.

```
ns-train lantern-nerfacto --help
```

If you need to use less frames, add this argument: `--pipeline.datamanager.train-num-images-to-sample-from 200 --pipeline.datamanager.eval-num-images-to-sample-from 30`

> Thing to keep in mind: the way Nerfstudio splits train and test images may not be optimal: left might be in one dataset and right in another... that's a bit of a data leak.

Examples

```
ns-train lantern-nerfacto --data data/lab_ground_floor/colmap_out_smart --pipeline.datamanager.eval-num-images-to-sample-from 0
```

## 3. Running MomoNet

(Using which frames?) A pretrained UNet image-to-image network called MomoNet will infill the missing dynamic range and store these predictions in new images.
It will be trained on the Laval indoor HDR dataset to do this task.

TODO: validate this part

## 4. Train from scratch or fine-tune the NeRF model using MomoNet's predictions



```
```