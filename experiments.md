

``
renders/lab_downstairs_ns/2024-05-10_211317/hdr-nerfacto
``

No camera optim

outputs/lab_downstairs_ns/hdr-nerfacto/2024-05-11_141957/config.yml

Train:
```
export SCENE=lab_downstairs_ns

ns-train hdr-nerfacto --data data/$SCENE --pipeline.datamanager.train-num-images-to-sample-from 400 --pipeline.datamanager.eval-num-images-to-sample-from 30
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



Test wo CRF

export SCENE=lab_downstairs_ns
export MODEL_NAME=hdr-nerfacto/2024-05-12_115558
