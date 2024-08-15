import os
import shutil
import json
import argparse

def move_subfolder(source_dir, destination_dir):
    if not os.path.exists(source_dir):
        print(f"Source directory '{source_dir}' does not exist.")
        return
    
    os.makedirs(destination_dir, exist_ok=True)

    try:
        for filename in os.listdir(source_dir):
            if filename.lower().count('png') < 2:
                shutil.move(os.path.join(source_dir, filename), os.path.join(destination_dir, filename))
    except Exception as e:
        print(f"Error moving subfolder: {e}")

    delete_empty_folder(source_dir)

def copy_subfolder(source_dir, destination_dir):
    if not os.path.exists(source_dir):
        print(f"Source directory '{source_dir}' does not exist.")
        return
    
    os.makedirs(destination_dir, exist_ok=True)

    try:
        for filename in os.listdir(source_dir):
            if filename.lower().count('png') < 2:
                shutil.copy2(os.path.join(source_dir, filename), os.path.join(destination_dir, filename))
    except Exception as e:
        print(f"Error copying subfolder: {e}")

def delete_empty_folder(folder_path):
    try:
        if os.path.exists(folder_path):
            if not os.listdir(folder_path):
                os.rmdir(folder_path)
                print(f"Folder '{folder_path}' deleted successfully.")
        else:
            print(f"Folder '{folder_path}' does not exist.")
    except Exception as e:
        print(f"Error deleting folder: {e}")

def create_capture_settings_file(file_path, shutter_speed):
    capture_settings = {
        "left": {
            "white_balance": 3500,
            "shutter_speed": shutter_speed,
            "iso": 800
        },
        "right": {
            "white_balance": 3500,
            "shutter_speed": 0.00004,
            "iso": 800
        }
    }

    try:
        with open(file_path, 'w') as file:
            json.dump(capture_settings, file, indent=4)
    except Exception as e:
        print(f"Error creating JSON file: {e}")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input_dir", type=str, default="/mnt/data/scene/")
    argparser.add_argument("--shutter_speed", type=float, default=0.003125)
    args = argparser.parse_args()

    dest_folder = args.input_dir
    masks_folder = args.input_dir + "masks/"

    for item in os.listdir(masks_folder):
        copy_subfolder(masks_folder + item, dest_folder + item + "/mask")

    delete_empty_folder(args.input_dir + "masks")

    create_capture_settings_file(dest_folder + "capture_settings.json", args.shutter_speed)




    


