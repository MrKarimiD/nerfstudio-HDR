import subprocess
import argparse

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input_dir", type=str, required=True, default="/mnt/data/scene/", help="Directory containing all the 4 videos")
    argparser.add_argument("--shutter_speed", type=float, required=True, default=0.003125)
    args = argparser.parse_args()

    subprocess.run(["python", "lantern_scripts/sync_and_trim.py", "--input_dir", args.input_dir])
    subprocess.run(["python", "lantern_scripts/mp4_videos_to_png_frames.py", "--input_dir", args.input_dir + "data/"])
    subprocess.run(["python", "lantern_scripts/mask_humans2.py", "--input_dir", args.input_dir + "data/", "--output_dir", args.input_dir + "data/masks"])
    subprocess.run(["python", "lantern_scripts/create_data_folder.py", "--input_dir", args.input_dir + "data/", "--shutter_speed", str(args.shutter_speed)])
    subprocess.run(["python", "lantern_scripts/create_sfm_folder.py", "--input_dir", args.input_dir])
