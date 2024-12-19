# How it works?

## 0. Setup

```
tmux new -s nerf_studio
docker run -it --rm -v /gel/usr/{USER_NAME}:/mnt/workspace/ -v /home-local2/{USER_NAME}.extra.nobkp/:/mnt/data/ --gpus '"device=0"' -p 8008:8008 -p 6006:6006 lantern_docker:latest /bin/bash
cd /mnt/workspace/lantern/nerfstudio-HDR

git clone https://github.com/MrKarimiD/nerfstudio-HDR.git
conda activate nerfstudio-HDR
export PYTHONPATH=$pwd:$PYTHONPATH
pip install -e .
pip install skylibs equilib scikit-surgerycore pydub
pip install -U git+https://github.com/luca-medeiros/lang-segment-anything.git
```

If there is an issue with equilib, do the following...
```
pip uninstall equilib
cd ../equilib
python setup.py develop
cd ../nerfstudio-HDR
```

## 1. Data acquisition & pre-processing

1. Setup the apparatus (stick with two Ricoh cameras)
    > Cameras:
    >
    > **Camera 1**: left, well-exposed, serial: YN14100695 \
    > **Camera 3**: right, fast-exposed (under-exposed), serial: YN14111000

    - To connect the other camera:
        - Turn wifi off
        - Close the app
        - Turn wifi on
        - Open the app
        - Connect the other camera
    - Camera setup:
        - Install Ricoh Theta app
        - Connect the camera to the app
        - Change camera parameters: select manual (bottom right)
            - First time: turn on "CT Settings" in Shooting settings
            - Apreture: 2.1
            - Shutter speed: determined below
            - ISO: 800
            - WB: 3500
    -  Make sure the cameras point up to avoid having lights in the seam.

2. First capture with well-exposed cameras.
    - Change shutter speed to get a well-exposed image. **Note the shutter speed.**
    - Start the video on one camera. Connect to the other camera following the steps above and start the video on the second camera.
    - Clap at the beginning and at the end to use for synchronization.
    - Go around the scene with the two cameras well-exposed.

3. Second capture with Camera 1 (left) well-exposed and Camera 2 (right) fast-exposed. Change shutter speed of Camera 2 to 1/25000 (fastest exposure) and repeat the steps above 

4. Import the files on computer. If on a mac, use Ricoh Theta File Transfer for Mac

5. Get the equirectangular videos by processing them with the Ricoh Theta computer app.
    - Drag and drop the video file into the app window.
    - Make sure both boxes are unchecked.
    - Name files as fallows
        - Camera 1:
            - First capture: left_sfm
            - Second capture: left_e1
        - Camera 2:
            - First capture: right_sfm
            - Second capture: right_e2

6. Transfer videos on lab machine.

    ```
    scp -r /path/to/source/file /path/to/destination/
    ```

7. Process the videos for OpenSFM. Input the right shutter speed noted above.

    ```
    python lantern_scripts/process_videos_for_sfm.py --input_dir /mnt/data/scene/ --shutter_speed 0.004
    ```

## 2. Process data using OpenSFM

```
bin/opensfm_run_all /mnt/data/scene/sfm/
```

To view results:
```
python3 viewer/server.py -d /mnt/data/scene/sfm/
```
Check that the camera positions make sense and that the number of clusters is low.

## 2. Process data for NeRF

```
ns-process-data lantern-openSFM  --data /mnt/data/scene/data/  --output-dir /mnt/data/scene/scene_ns/ --metadata  /mnt/data/scene/sfm/reconstruction.json
```

For processing existing dataset that used colmap for camera positions:
```
ns-process-data images --data /mnt/data/garden/images --output-dir /mnt/data/garden_processed --skip-colmap --skip-image-processing --colmap-model-path /mnt/data/garden/sparse/0
```

## 3. Train NeRF

1. Run step 1 of lantern-nerfacto.

    ```
    ns-train lantern-nerfacto --data /mnt/data/scene/scene_ns/ --viewer.websocket-port 8008 --pipeline.datamanager.train-num-images-to-sample-from 1800 --pipeline.model.lantern_steps 1 --pipeline.datamanager.pixel-sampler.lantern_steps 1 --max-num-iterations 60000
    ```

    1.1. For better alignment between the well and the fast exposed images, run the following command.
    ```
    python lantern_scripts/align_fast_to_well.py --input_dir /mnt/data/scene/scene_ns/ --config outputs/scene_ns/lantern-nerfacto/2024-08-07_143013/config.yml
    ```

2. Run step 2 of lantern-nerfacto.
    
    ```
    ns-train lantern-nerfacto --data /mnt/data/scene/scene_ns/ --viewer.websocket-port 8008 --pipeline.datamanager.train-num-images-to-sample-from 1800 --pipeline.model.lantern_steps 2 --pipeline.datamanager.pixel-sampler.lantern_steps 2 --load-dir /mnt/workspace/lantern/nerfstudio-HDR/outputs/scene_ns/lantern-nerfacto/2024-06-10_155559/nerfstudio_models --pipeline.model.apply_mu_law False
    ```

## 4. View results

To use the interface:
```
ns-viewer --load-config outputs/scene_ns/lantern-nerfacto/2024-06-06_191338/config.yml --viewer.websocket-port 8008
```
- For the right side up:
    - Go to SCENE sub-menu.
    - Clic on RESET UP DIRECTION.
- To render a video with the interface command: 
    - Go to RENDER sub-menu.
    - Clic on ADD CAMERA to manualy set camera view points.
    - Clic on RENDER to get the command.
        ```
        ns-render camera-path --load-config outputs/scene_ns/lantern-nerfacto/2024-06-06_191338/config.yml --camera-path-filename /mnt/data/scene/scene_ns/camera_paths/2024-06-06_191338.json --output-path /mnt/data/scene/renders/2024-06-06_191338.mp4
        ```
    - To render a video fast-exposed: add --rendered-output-names rgb_fast
        ```
        ns-render camera-path --load-config outputs/scene_ns/lantern-nerfacto/2024-06-06_191338/config.yml --camera-path-filename /mnt/data/scene/scene_ns/camera_paths/2024-06-06_191338.json --output-path /mnt/data/scene/renders/2024-06-06_191338.mp4 --rendered-output-names rgb_fast
        ```

To get evaluation images (well-exposed):
```
ns-eval --load-config=outputs/scene_ns/lantern-nerfacto/2024-06-17_152246/config.yml --output-path=output.json --render_output_path=/mnt/data/scene/eval
```

- To get fast-exposed evaluation images, change these two lines in nerfstudio/lantern/model.py:
    ```
    line 238: if batch["exposure"] != 1.0: # change to == for testing fast-exposed
    line 242: return None, None # comment for testing fast-exposed
    ```

- To get evaluation images and tensorboard:
    ```
    ns-eval --load-config=outputs/scene_ns/lantern-nerfacto/2024-06-20_142413/config.yml --output-path=output.json --render_output_path=/mnt/data/scene/eval  --vis viewer+tensorboard
    ```
- To load tensorboard results:
    ```
    tensorboard --logdir=/path/to/file
    ```
    - Connect to http://localhost:6006/

## 4. Calculate metrics

1. Get ground truth transformations:
    ```
    ns-process-data lantern-GT-HDR --data /mnt/data/lab_downstairs_ns/ --output-dir /mnt/data/metrics/lab_downstairs --metadata /mnt/data/lab_downstairs_ns/reconstruction.json --checkpoint /mnt/workspace/lantern/nerfstudio-HDR/outputs/lab_downstairs_ns/lantern-nerfacto/2024-10-06_232947/
    ```

2. Render panos at ground truth positions for well and fast exposed:
    ```
    ns-render camera-path --load-config /mnt/workspace/lantern/nerfstudio-HDR/outputs/lab_downstairs_ns/lantern-nerfacto/2024-10-06_232947/config.yml --camera-path-filename /mnt/data/metrics/lab_downstairs/GT_transforms.json --output-path /mnt/data/metrics/renders/lab_downstairs_aligned --output-format images --rendered_output_names
    ```
    ```
    ns-render camera-path --load-config /mnt/workspace/lantern/nerfstudio-HDR/outputs/lab_downstairs_ns/lantern-nerfacto/2024-10-06_232947/config.yml --camera-path-filename /mnt/data/metrics/lab_downstairs/GT_transforms.json --output-path /mnt/data/metrics/renders/lab_downstairs_aligned_fast --output-format images --rendered_output_names rgb_fast
    ```

3. Combine both well and fast exposed panos:
    ```
    python lantern_scripts/combine.py --well_dir /mnt/data/metrics/renders/lab_downstairs_aligned/ --fast_dir /mnt/data/metrics/renders/lab_downstairs_aligned_fast/ --out_dir /mnt/data/metrics/renders/lab_downstairs_lin/ --experiment_location /mnt/data/lab_downstairs_ns/ --do_linearization
    ```

4. Make sure the panos from renders have the same names as the GT panos.

5. Calculate PSNR, SSIM and LPIPS:
    ```
    python lantern_scripts/ldr_res.py --gt_dir /mnt/data/metrics/GT_lab_downstairs/pano/  --data_dir /mnt/data/metrics/renders/lab_downstairs_lin/
    ```

5. Generate HDR renders with blender:
    ```
    blender --background /mnt/data/metrics/scene_new.blend -P lantern_scripts/hdr_blender.py -- /mnt/data/metrics/renders/lab_downstairs_lin_less/ /mnt/data/metrics/renders/lab_downstairs_lin_renders/
    ```

6. Calculate si-RMSE, RMSE, RGB ang. and PSNR (LDR r.). Need to change values in code:
    ```
    python lantern_scripts/res_table_1.py
    ```

7. Convert exr panos to hdr:
    ```
    convert GT1.exr GT1.hdr
    ```

8. Calulate PU-PSNR, HDR-VDP and PU-SSIM following the steps in the following link:
    https://github.com/darthgera123/PanoHDR-NeRF/tree/main/LANet/metrics
    ```
    cd hdrvdp-3.0.6
    ```
    ```
    matlab -batch "metric('../../data/GT_meeting_room/', '../../data/results_meeting_room/')"
    ```

    ```
    cd ../pu21/matlab/examples
    ```
    ```
    matlab -batch "pupsnr('../../../../data/GT_meeting_room/', '../../../../data/results_meeting_room/')"
    ```