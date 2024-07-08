# How it works?

## 0. Setup

```
export PYTHONPATH=$pwd:$PYTHONPATH
pip install skylibs equilib
```

## 1. Data acquisition & pre-processing

1. Acquire data using the apparatus (stick with two Ricoh cameras)

    Make sure the cameras point up to avoid having lights in the seam.

    Do two recordings with both Ricoh Theta cameras.

    Camera setup:
        Install Ricoh Theta app
        Connect the camera to the app
        Change camera parameters: select manual (bottom right)
            First time: turn on "CT Settings" in Shooting settings
            Apreture: 2.1
            Shutter speed: determined below
            ISO: 800
            WB: 3500
    
    To connect the other camera:
        Turn wifi off
        Close the app
        Turn wifi on
        Open the app
        Connect the other camera

    > Cameras:
    > 
    > **camera 1**: left, well-exposed, serial: YN14100695 \
    > **camera 3**: right, under-exposed, serial: YN14111000

2. Take one video capture with the two cameras well-exposed. IMPORTANT: Note the shutter speed!!!

    Change shutter speed to get a well exposed image.
    Clap at the beginning and at the end to use for synchronization.

3. Take one video capture with Camera 1 (left) well-exposed and Camera 2 (right) fast-exposed.
    
    Change shutter speed of Camera 2 to 1/25000 (fastest exposure).
    Clap at the beginning and at the end to use for synchronization.

4. Import the files on computer

    If on mac, use Ricoh Theta File Transfer for Mac

5. Get the equirectangular videos by processing them with the Ricoh Theta computer app.

    Drag and drop the video file into the app window.
    Make sure both boxes are unchecked.
    Name files as fallows
    Camera 1:
        First capture: left_sfm
        Second capture: left_e1
    Camera 2:
        First capture: right_sfm
        Second capture: right_e2

6. Transfer videos on lab machine.

    ```
    scp -r /path/to/source/file /path/to/destination/
    ```

7. Process the videos for OpenSFM.

    ```
    python lantern_scripts/process_videos_for_sfm.py --input_dir /mnt/data/small_scene_window/ --shutter_speed 0.004
    ```

## 2. Process data using OpenSFM

```
bin/opensfm_run_all /mnt/data/scene/sfm/
```

## 2. Process data for NeRF

```
ns-process-data lantern-openSFM  --data /mnt/data/small_scene/data/  --output-dir /mnt/data/small_scene/ns/ --metadata  /mnt/data/small_scene/sfm/reconstruction.json
```

## 3. Train NeRF

1. Run step 1 of lantern-nerfacto.

    ```
    ns-train lantern-nerfacto --data /mnt/data/meeting_room_small_ns_no_exr/ --viewer.websocket-port 8008 --pipeline.datamanager.train-num-images-to-sample-from 1800 --pipeline.model.lantern_steps 1 --pipeline.datamanager.pixel-sampler.lantern_steps 1 --max-num-iterations 60000
    ```

2. Run step 2 of lantern-nerfacto.

    ```
    ns-train lantern-nerfacto --data /mnt/data/meeting_room_small_ns/ --viewer.websocket-port 8008 --pipeline.datamanager.train-num-images-to-sample-from 1800 --pipeline.model.lantern_steps 2 --pipeline.datamanager.pixel-sampler.lantern_steps 2 --load-dir /mnt/workspace/lantern/nerfstudio-HDR/outputs/meeting_room_small_ns/lantern-nerfacto/2024-06-10_155559/nerfstudio_models --pipeline.model.apply_mu_law False
    ```

## 4. View results

Use the interface :
```
ns-viewer --load-config outputs/tree_hill/nerfacto/2024-06-06_191338/config.yml --viewer.websocket-port 8008
```

Render a video with the interface command:
Move around and add cameras to manualy set a camera path.
```
ns-render camera-path --load-config outputs/tree_hill/nerfacto/2024-06-06_191338/cclearonfig.yml --camera-path-filename /mnt/data/tree_hill/camera_paths/2024-06-06_191338.json --output-path renders/tree_hill/2024-06-06_191338.mp4
```

Get evaluation images:
```
ns-eval --load-config=outputs/meeting_room_ns_no_exr/nerfacto/2024-06-17_152246/config.yml --output-path=output.json --render_output_path=/mnt/data/temp_folder_meeting_room_unaligned
```

To get evaluation images and tensorboard:
```
ns-eval --load-config=outputs/meeting_room_ns_no_exr/lantern-nerfacto/2024-06-20_142413/config.yml --output-path=output.json --render_output_path=/mnt/data/temp_folder_meeting_room_large_dataset_2  --vis viewer+tensorboard
```

Load tensorboard results:
```
tensorboard --logdir=Downloads/images_for_pts/test/runs/
```
And connect to http://localhost:6006/
