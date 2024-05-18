#!/bin/bash
echo "Evaluation for linearization then nerf pipeline"

dataset_name=meeting_room_ns
method_name=lantern-nerfacto
root_dir=/mnt/data/nerfstudio_ds/real_data/meeting_room_openSFM
dataset_dir=$root_dir/$dataset_name/
opensfm_dir=$root_dir/meeting_room_openSFM
opensfm_file=$opensfm_dir/reconstruction.json

echo "Step one of training"
ns-train lantern-nerfacto --data $dataset_dir --viewer.websocket-port 8008 --pipeline.datamanager.train-num-images-to-sample-from 1800 \
--pipeline.model.lantern_steps 1 --pipeline.datamanager.pixel-sampler.lantern_steps 1 --pipeline.model.apply_mu_law False \
--pipeline.datamanager.camera-optimizer.mode off --viewer.quit-on-train-completion True --output-dir outputs --timestamp ablation_step_1

echo "Step two of training"
ns-train lantern-nerfacto --data $dataset_dir --viewer.websocket-port 8008 --pipeline.datamanager.train-num-images-to-sample-from 1800 \
--pipeline.model.lantern_steps 2 --pipeline.datamanager.pixel-sampler.lantern_steps 2 --pipeline.model.apply_mu_law False \
--pipeline.datamanager.camera-optimizer.mode off --viewer.quit-on-train-completion True --load-dir outputs/$dataset_name/lantern-nerfacto/ablation_step_1/nerfstudio_models \
--output-dir outputs --timestamp ablation_step_2

# checkpoint_dir=outputs/$dataset_name/lantern-nerfacto/ablation_step_2/
# output_dir=renders/$dataset_name/$method_name/ablation_linearization_then_nerf

# echo "Creating the test set"
# ns-process-data lantern-GT-HDR --data $dataset_dir --output-dir $dataset_dir --metadata $opensfm_file --checkpoint $checkpoint_dir

# echo "Rendering for the well exposed"
# ns-render camera-path --load-config $checkpoint_dir/config.yml --camera-path-filename $dataset_dir/GT_transforms.json --output-path $output_dir/well-expo/  --output-format images

# echo "Rendering for the fast exposure"
# ns-render camera-path --load-config $checkpoint_dir/config.yml --camera-path-filename $dataset_dir/GT_transforms.json --output-path $output_dir/fast-expo/  --output-format images --rendered_output_names rgb_fast

# #conda init bash
# conda activate everlight_new

# echo "Combine two exposures"
# python lantern_scripts/combine.py --well_dir $output_dir/well-expo/ --fast_dir $output_dir/fast-expo/ --out_dir $output_dir/pano/ --experiment_location $root_dir

# conda activate nerfstudio-HDR

# blender --background lantern_scripts/scene_new.blend -P lantern_scripts/hdr_blender.py -- $output_dir/pano/ $output_dir/render/