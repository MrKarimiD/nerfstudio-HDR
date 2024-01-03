import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import argparse
from tqdm import tqdm


# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='', help='directory to data')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--stick',type=str,default='')
    args = parser.parse_args()
    out = f'{args.output_dir}/mask/'
    # out = f'{args.output_dir}/overlay'
    if(not os.path.isdir(out)):
        os.makedirs(out)
    # stick = cv2.imread(args.stick)
    imgs = os.listdir(os.path.join(args.input_dir,'ldr'))
    # imgs = os.listdir(args.input_dir)
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    predictor = DefaultPredictor(cfg)
    for file in tqdm(imgs):
        path = f'{args.input_dir}/ldr/{file}'
        img = cv2.imread(path)
        panoptic_seg, segments_info = predictor(img)["panoptic_seg"]
                # category_id = 0 is person which we need. the number in panoptic_seg is the corresponding id not category id
        new_seg_info=[]
        nimg = panoptic_seg.cpu().numpy()
        w,h = nimg.shape
        mask = np.ones(nimg.shape)*255
        for seg in segments_info:
            if seg['category_id']==0:
                new_seg_info.append(seg)
                mask[nimg == seg['id']] = 0
        # v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        # out_img = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), new_seg_info)
        # im = out_img.get_image()[:, :, ::-1]
        # print(im.shape)
        # mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
        fmask = np.ones((w,h,3))
        # for i in range(0,3):
        #     fmask[:,:,i] = mask
        fmask[:,:,:] = mask.reshape(w,h,1)
        # print(fmask.shape,stick.shape)
        # fmask = fmask & stick.astype('uint8')
        # kernel = np.ones((5, 5), np.uint8)
        # fmask = cv2.erode(fmask,kernel)
        cv2.imwrite(f'{out}/{file}',fmask)
        # print(out)
        # print(file)
        # output_path = os.path.join(out,file)
        # cv2.imwrite(output_path,im)
