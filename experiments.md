

``
renders/lab_downstairs_ns/2024-05-10_211317/hdr-nerfacto
``

No camera optim

outputs/lab_downstairs_ns/hdr-nerfacto/2024-05-11_141957/config.yml

Train:
```
export SCENE=lab_downstairs_ns

ns-train hdr-nerfacto --data data/$SCENE --pipeline.datamanager.train-num-images-to-sample-from 1800
```

Eval (GT+normal):
```
export SCENE=lab_downstairs_ns
export MODEL_NAME=hdr-nerfacto/2024-05-11_141957

ns-eval --load-config outputs/$SCENE/$MODEL_NAME/config.yml --output-path ablations_out/$SCENE/$MODEL_NAME/output.json --render-output-path ablations_out/$SCENE/$MODEL_NAME ; \
ns-render camera-path --load-config outputs/$SCENE/$MODEL_NAME/config.yml --camera-path-filename data/$SCENE/GT_transforms.json --output-path renders/$SCENE/${MODEL_NAME}_well --output-format images --image-format png --rendered-output-names rgb ; \
ns-render camera-path --load-config outputs/$SCENE/$MODEL_NAME/config.yml --camera-path-filename data/$SCENE/GT_transforms.json --output-path renders/$SCENE/${MODEL_NAME}_fast --output-format images --image-format png --rendered-output-names rgb_fast ; \
ns-render camera-path --load-config outputs/$SCENE/$MODEL_NAME/config.yml --camera-path-filename data/$SCENE/GT_transforms.json --output-path renders/$SCENE/${MODEL_NAME}_hdr --output-format images --image-format exr --rendered-output-names rgb_hdr
```

Move (outside docker)
```
mkdir -p /gel/usr/mokad6/ml_class/fred-out/ ; \
cp -R /home-local2/frfoc1.extra.nobkp/nerfstudio-HDR/ablations_out /gel/usr/mokad6/ml_class/fred-out/ns-eval ; \
cp -R /home-local2/frfoc1.extra.nobkp/nerfstudio-HDR/renders /gel/usr/mokad6/ml_class/fred-out/ns-render
```

1800 samples:
export SCENE=lab_downstairs_ns
export MODEL_NAME=hdr-nerfacto/2024-05-12_115558

1600 samples:
export SCENE=JF_office_ns
export MODEL_NAME=hdr-nerfacto/2024-05-14_152107




Test wo CRF

export SCENE=lab_downstairs_ns
export MODEL_NAME=hdr-nerfacto/2024-05-12_115558



## Fix pixel sampler

1800 samples, fix pixel sampler:
export SCENE=Henry_Gagnon_blue_et_rouge_ns
export MODEL_NAME=hdr-nerfacto/2024-05-14_175753
Exp: 0.002.....
DONE

1800 samples, fix pixel sampler:
export SCENE=JF_office_ns
export MODEL_NAME=hdr-nerfacto/2024-05-14_182719
Exp: 0.008

1800 samples, fix pixel sampler:
export SCENE=lab_downstairs_ns
export MODEL_NAME=hdr-nerfacto/2024-05-14_191020
Exp: 0.01

1800 samples, fix pixel sampler:
export SCENE=meeting_room_ns
export MODEL_NAME=hdr-nerfacto/2024-05-15_203121
Exp: 0.008

1800 samples, w/o CRF, fix pixel sampler:
export SCENE=lab_downstairs_ns
export MODEL_NAME=hdr-nerfacto-wo-crf/2024-05-14_194316
Exp: 0.01

1800 samples, w/o CRF, clip before acc, fix pixel sampler:
export SCENE=lab_downstairs_ns
export MODEL_NAME=hdr-nerfacto-wo-crf-clip-bf-acc/2024-05-14_200611
Exp: 0.01


## Export all

export SCENE=Henry_Gagnon_blue_et_rouge_ns ; \
export MODEL_NAME=hdr-nerfacto/2024-05-14_175753 ; \
ns-eval --load-config outputs/$SCENE/$MODEL_NAME/config.yml --output-path ablations_out/$SCENE/$MODEL_NAME/output.json --render-output-path ablations_out/$SCENE/$MODEL_NAME ; \
export SCENE=JF_office_ns ; \
export MODEL_NAME=hdr-nerfacto/2024-05-14_182719 ; \
ns-eval --load-config outputs/$SCENE/$MODEL_NAME/config.yml --output-path ablations_out/$SCENE/$MODEL_NAME/output.json --render-output-path ablations_out/$SCENE/$MODEL_NAME ; \
export SCENE=lab_downstairs_ns ; \
export MODEL_NAME=hdr-nerfacto/2024-05-14_191020 ; \
ns-eval --load-config outputs/$SCENE/$MODEL_NAME/config.yml --output-path ablations_out/$SCENE/$MODEL_NAME/output.json --render-output-path ablations_out/$SCENE/$MODEL_NAME ; \
export SCENE=lab_downstairs_ns ; \
export MODEL_NAME=hdr-nerfacto-wo-crf/2024-05-14_194316 ; \
ns-eval --load-config outputs/$SCENE/$MODEL_NAME/config.yml --output-path ablations_out/$SCENE/$MODEL_NAME/output.json --render-output-path ablations_out/$SCENE/$MODEL_NAME ; \
export SCENE=lab_downstairs_ns ; \
export MODEL_NAME=hdr-nerfacto-wo-crf-clip-bf-acc/2024-05-14_200611 ; \
ns-eval --load-config outputs/$SCENE/$MODEL_NAME/config.yml --output-path ablations_out/$SCENE/$MODEL_NAME/output.json --render-output-path ablations_out/$SCENE/$MODEL_NAME

### Models trained for 60K steps

outputs/lab_downstairs_ns/hdr-nerfacto/2024-05-18_012547/config.yml
outputs/meeting_room_ns/hdr-nerfacto/2024-05-18_020406/config.yml

fast_exp=0.01
export SCENE=lab_downstairs_ns ; \
export MODEL_NAME=hdr-nerfacto/2024-05-18_012547 ; \
ns-render camera-path --load-config outputs/$SCENE/$MODEL_NAME/config.yml --camera-path-filename data/$SCENE/GT_transforms.json --output-path renders/$SCENE/${MODEL_NAME}_well --output-format images --image-format png --rendered-output-names rgb ; \
ns-render camera-path --load-config outputs/$SCENE/$MODEL_NAME/config.yml --camera-path-filename data/$SCENE/GT_transforms.json --output-path renders/$SCENE/${MODEL_NAME}_fast --output-format images --image-format png --rendered-output-names rgb_fast ; \
ns-render camera-path --load-config outputs/$SCENE/$MODEL_NAME/config.yml --camera-path-filename data/$SCENE/GT_transforms.json --output-path renders/$SCENE/${MODEL_NAME}_hdr --output-format images --image-format exr --rendered-output-names rgb_hdr

export SCENE=meeting_room_ns ; \
export MODEL_NAME=hdr-nerfacto/2024-05-18_020406 ; \
ns-render camera-path --load-config outputs/$SCENE/$MODEL_NAME/config.yml --camera-path-filename data/$SCENE/GT_transforms.json --output-path renders/$SCENE/${MODEL_NAME}_well --output-format images --image-format png --rendered-output-names rgb ; \
ns-render camera-path --load-config outputs/$SCENE/$MODEL_NAME/config.yml --camera-path-filename data/$SCENE/GT_transforms.json --output-path renders/$SCENE/${MODEL_NAME}_fast --output-format images --image-format png --rendered-output-names rgb_fast ; \
ns-render camera-path --load-config outputs/$SCENE/$MODEL_NAME/config.yml --camera-path-filename data/$SCENE/GT_transforms.json --output-path renders/$SCENE/${MODEL_NAME}_hdr --output-format images --image-format exr --rendered-output-names rgb_hdr

fast_exp=0.004
export SCENE=my_apartment_ns ; \
export MODEL_NAME=hdr-nerfacto/2024-05-18_133026 ; \
ns-render camera-path --load-config outputs/$SCENE/$MODEL_NAME/config.yml --camera-path-filename data/$SCENE/GT_transforms.json --output-path renders/$SCENE/${MODEL_NAME}_well --output-format images --image-format png --rendered-output-names rgb ; \
ns-render camera-path --load-config outputs/$SCENE/$MODEL_NAME/config.yml --camera-path-filename data/$SCENE/GT_transforms.json --output-path renders/$SCENE/${MODEL_NAME}_fast --output-format images --image-format png --rendered-output-names rgb_fast ; \
ns-render camera-path --load-config outputs/$SCENE/$MODEL_NAME/config.yml --camera-path-filename data/$SCENE/GT_transforms.json --output-path renders/$SCENE/${MODEL_NAME}_hdr --output-format images --image-format exr --rendered-output-names rgb_hdr

fast_exp=0.004
export SCENE=LVSN_LAB_OpenSFM
export MODEL_NAME=hdr-nerfacto/2024-05-18_143132
ns-render camera-path --load-config outputs/$SCENE/$MODEL_NAME/config.yml --camera-path-filename data/$SCENE/GT_transforms.json --output-path renders/$SCENE/${MODEL_NAME}_well --output-format images --image-format png --rendered-output-names rgb ; \
ns-render camera-path --load-config outputs/$SCENE/$MODEL_NAME/config.yml --camera-path-filename data/$SCENE/GT_transforms.json --output-path renders/$SCENE/${MODEL_NAME}_fast --output-format images --image-format png --rendered-output-names rgb_fast ; \
ns-render camera-path --load-config outputs/$SCENE/$MODEL_NAME/config.yml --camera-path-filename data/$SCENE/GT_transforms.json --output-path renders/$SCENE/${MODEL_NAME}_hdr --output-format images --image-format exr --rendered-output-names rgb_hdr


### Ablations 60K


export SCENE=lab_downstairs_ns ; \
export MODEL_NAME=hdr-nerfacto-wo-crf/2024-05-18_171020
ns-render camera-path --load-config outputs/$SCENE/$MODEL_NAME/config.yml --camera-path-filename data/$SCENE/GT_transforms.json --output-path renders/$SCENE/${MODEL_NAME}_hdr --output-format images --image-format exr --rendered-output-names rgb_hdr

export SCENE=lab_downstairs_ns ; \
export MODEL_NAME=hdr-nerfacto-wo-crf-clip-bf-acc/2024-05-18_174929
ns-render camera-path --load-config outputs/$SCENE/$MODEL_NAME/config.yml --camera-path-filename data/$SCENE/GT_transforms.json --output-path renders/$SCENE/${MODEL_NAME}_hdr --output-format images --image-format exr --rendered-output-names rgb_hdr
