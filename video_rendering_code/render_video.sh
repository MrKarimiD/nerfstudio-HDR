
export scene_name=my_apartment_ns
export output_dir=renders/${scene_name}/lantern-nerfacto/video_images
export root_dir=data/${scene_name}
export model_path=outputs/${scene_name}/lantern-nerfacto/2024-05-16_212421/config.yml

# camera paths
# video ground: data/Henry_Gagnon_blue_et_rouge_ns/camera_paths/2024-05-16_015020.json
# still bench: 2024-05-16_015020_render
# pespective cam: 2024-05-16_015020_render3

# meeting room
# export camera_path_name=2024-05-16_201826_persp
# export camera_path_name=2024-05-16_201826_envmap

# my_apartment
# export camera_path_name=2024-05-16_212421_persp
# export camera_path_name=2024-05-16_212421_env1
# export camera_path_name=2024-05-16_212421_env2
# export camera_path_name=2024-05-16_212421_env3
# export camera_path_name=2024-05-16_212421_env4
export camera_path_name=2024-05-16_212421_env5
export camera_path=data/${scene_name}/camera_paths/${camera_path_name}.json

ns-render camera-path --load-config $model_path --camera-path-filename $camera_path --output-path $output_dir/${camera_path_name}/well-expo/  --output-format images
ns-render camera-path --load-config $model_path --camera-path-filename $camera_path --output-path $output_dir/${camera_path_name}/fast-expo/  --output-format images  --rendered_output_names rgb_fast

echo "Combine two exposures"
python video_rendering_code/combine.py --well_dir $output_dir/${camera_path_name}/well-expo/ --fast_dir $output_dir/${camera_path_name}/fast-expo/ --out_dir $output_dir/${camera_path_name}/combined/ --do_linearization --experiment_location $root_dir

# blender --background video_rendering_code/scene_new.blend -P video_rendering_code/hdr_blender.py -- $output_dir/${camera_path_name}/combined/ $output_dir/${camera_path_name}/render/




#########

export camera_path_name=2024-05-16_212421_env1
export camera_path=data/${scene_name}/camera_paths/${camera_path_name}.json

echo "Combine two exposures"
python video_rendering_code/combine.py --well_dir $output_dir/${camera_path_name}/well-expo/ --fast_dir $output_dir/${camera_path_name}/fast-expo/ --out_dir $output_dir/${camera_path_name}/combined/ --do_linearization --experiment_location $root_dir


export camera_path_name=2024-05-16_212421_env2
export camera_path=data/${scene_name}/camera_paths/${camera_path_name}.json

echo "Combine two exposures"
python video_rendering_code/combine.py --well_dir $output_dir/${camera_path_name}/well-expo/ --fast_dir $output_dir/${camera_path_name}/fast-expo/ --out_dir $output_dir/${camera_path_name}/combined/ --do_linearization --experiment_location $root_dir




export camera_path_name=2024-05-16_212421_env3
export camera_path=data/${scene_name}/camera_paths/${camera_path_name}.json

echo "Combine two exposures"
python video_rendering_code/combine.py --well_dir $output_dir/${camera_path_name}/well-expo/ --fast_dir $output_dir/${camera_path_name}/fast-expo/ --out_dir $output_dir/${camera_path_name}/combined/ --do_linearization --experiment_location $root_dir




export camera_path_name=2024-05-16_212421_env4
export camera_path=data/${scene_name}/camera_paths/${camera_path_name}.json

echo "Combine two exposures"
python video_rendering_code/combine.py --well_dir $output_dir/${camera_path_name}/well-expo/ --fast_dir $output_dir/${camera_path_name}/fast-expo/ --out_dir $output_dir/${camera_path_name}/combined/ --do_linearization --experiment_location $root_dir
export camera_path_name=2024-05-16_212421_env5
export camera_path=data/${scene_name}/camera_paths/${camera_path_name}.json

echo "Combine two exposures"
python video_rendering_code/combine.py --well_dir $output_dir/${camera_path_name}/well-expo/ --fast_dir $output_dir/${camera_path_name}/fast-expo/ --out_dir $output_dir/${camera_path_name}/combined/ --do_linearization --experiment_location $root_dir
