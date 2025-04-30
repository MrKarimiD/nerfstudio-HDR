import os
import shutil
import re
import argparse

def copy_files(input_dir):
    brackets = os.listdir(input_dir)

    target_exr = os.path.join(input_dir, "GT_exr")
    target_jpg = os.path.join(input_dir, "GT_jpg")
    os.makedirs(target_exr, exist_ok=True)
    os.makedirs(target_jpg, exist_ok=True)

    for folder in brackets:
        if not folder.startswith("GT_"):
            folder_path = os.path.join(input_dir, folder)
            if os.path.isdir(folder_path):
                smallest_value = None
                smallest_file = None
                for file in os.listdir(folder_path):
                    if file.startswith("GT") and file.lower().endswith(".exr"):
                        source_file = os.path.join(folder_path, file)
                        destination_file = os.path.join(target_exr, file)
                        shutil.copy2(source_file, destination_file)
                        print(f"Copied EXR: {source_file} -> {destination_file}")

                    if file.lower().endswith(".jpg"):
                        numbers = re.findall(r'\d+', file)
                        num = int(numbers[0])
                        if smallest_value is None or num < smallest_value:
                            smallest_value = num
                            smallest_file = file

                source_file = os.path.join(folder_path, smallest_file)
                destination_file = os.path.join(target_jpg, f"{folder}.jpg")
                shutil.copy2(source_file, destination_file)
                print(f"Copied JPG: {source_file} -> {destination_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True,
                        help='GT directory containing all the subdirectories (e.g., GT1, GT2, ...) with frames')
    args = parser.parse_args()

    copy_files(args.input_dir)
