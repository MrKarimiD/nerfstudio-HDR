#!/bin/bash
echo "Evaluation for linearization then nerf pipeline"

dataset_name=LVSN_LAB_OpenSFM
method_name=lantern-nerfacto
root_dir=/mnt/data/nerfstudio_ds/real_data/lab_ground_floor_wb_3500/openSFM_full_lvsn
dataset_dir=$root_dir/$dataset_name/

echo "Step one of training"
# ns-train lantern-nerfacto --data $dataset_dir --viewer.websocket-port 8008 --pipeline.datamanager.train-num-images-to-sample-from 1800 \
# --pipeline.model.lantern_steps 1 --pipeline.datamanager.pixel-sampler.lantern_steps 1 --pipeline.model.apply_mu_law False \
# --pipeline.datamanager.camera-optimizer.mode off --viewer.quit-on-train-completion True --output-dir outputs --timestamp step_1

echo "Step two of training"
ns-train lantern-nerfacto --data $dataset_dir --viewer.websocket-port 8008 --pipeline.datamanager.train-num-images-to-sample-from 1800 \
--pipeline.model.lantern_steps 2 --pipeline.datamanager.pixel-sampler.lantern_steps 2 --pipeline.model.apply_mu_law False \
--pipeline.datamanager.camera-optimizer.mode off --viewer.quit-on-train-completion True --load-dir outputs/$dataset_name/lantern-nerfacto/step_1/nerfstudio_models \
--output-dir outputs --timestamp step_2

#  ns-train lantern-nerfacto --data $dataset_dir --viewer.websocket-port 8008 --pipeline.datamanager.train-num-images-to-sample-from 1800 \
#  --pipeline.model.lantern_steps 2 --pipeline.datamanager.pixel-sampler.lantern_steps 2 \
#  --pipeline.datamanager.camera-optimizer.mode off --load-dir outputs/$dataset_name/lantern-nerfacto/step_1/nerfstudio_models \
#  --output-dir outputs --timestamp step_2