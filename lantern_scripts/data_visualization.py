from os import makedirs
from html4vision import Col, imagetable
import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/mnt/data/nerfstudio_ds/real_data/coffeeroom_1102H/GT/GT_exr')
    args = parser.parse_args()

    cols = [
        Col('id1', 'ID'), # 1-based indexing
        Col('img', 'Left Pano', args.input_dir + 'left_e1/output_*.png'),
        Col('img', 'Left Mask', args.input_dir + 'masks/left_e1/output_*.png.masked.png'),
        Col('img', 'Right Pano', args.input_dir + 'right_e2/output_*.png'),
        Col('img', 'Right Mask', args.input_dir + 'masks/right_e2/output_*.png.masked.png'),
    ]

    # imagetable(cols, 'another-dir/pathrep.html', 'Path Replace Example', pathrep=('/mnt/data', '/home-local2/mokad6.extra.nobkp'))
    imagetable(cols, args.input_dir + 'data_index.html', 'NeRF sequence', pathrep=(args.input_dir, './'), imscale=0.1, zebra=True, sticky_header=True, overlay_toggle=True)