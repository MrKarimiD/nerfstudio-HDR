import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import json
import subprocess
import argparse

class ContourTooSmallException(Exception):
    pass

class Image:
    def __init__(self, image, threshold = 5):
        self.image = image.copy()
        self.threshold = threshold
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, self.thresholded = cv2.threshold(self.gray, threshold, 255, cv2.THRESH_BINARY)
        self.contours, _ = cv2.findContours(self.thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

class Contour:
    def __init__(self, contour, well_image_threshold_index = None, min_area = 50.0):
        self.contour = contour
        self.area = cv2.contourArea(contour)
        if self.area < min_area:
            raise ContourTooSmallException(f"Contour area {self.area} is too small.")
        self.centroid = np.mean(contour, axis=0)[0]
        self.relative_position = None
        self.match = None
        self.score = float('inf')
        self.well_image_threshold_index = well_image_threshold_index

    def overlap(self, contour):
        image_shape = (960, 960)
        mask1 = np.zeros(image_shape, dtype=np.uint8)
        mask2 = np.zeros(image_shape, dtype=np.uint8)
        cv2.drawContours(mask1, [self.contour], -1, 255, thickness=cv2.FILLED)
        cv2.drawContours(mask2, [contour.contour], -1, 255, thickness=cv2.FILLED)
        overlap_mask = cv2.bitwise_and(mask1, mask2)
        overlap_area = np.sum(overlap_mask) / 255
        return overlap_area > 0
    
    def get_matching_score(self, contour):
        dist_shape = cv2.matchShapes(self.contour, contour.contour, cv2.CONTOURS_MATCH_I2, 0.0)
        dist_spatial = np.linalg.norm(np.array(self.centroid) - np.array(contour.centroid))
        return 0.7 * dist_shape + 0.3 * dist_spatial
    
    def match_contour(self, contour, score):
        self.match = contour
        self.score = score
    
    def unmatch_contour(self):
        self.match = None
        self.score = float('inf')

    def get_points(self):
        return self.contour[:, 0, :]


class AlignImages:
    def __init__(self, well_image, fast_image):
        self.fast_image = Image(fast_image)
        self.fast_contours = []
        for contour in self.fast_image.contours:
            try:
                contour_obj = Contour(contour)
                self.fast_contours.append(contour_obj)
            except ContourTooSmallException as e:
                continue
        self.matched_well_contours = [None] * len(self.fast_contours)
        self.well_image = Image(well_image)
        self.well_images = []
        self.well_thresholds = (230, 235, 240, 245, 250)
        for threshold in self.well_thresholds:
            self.well_images.append(Image(well_image, threshold))
        self.image_height, self.image_width = self.fast_image.image.shape[:2]

    def all_contours_matched(self):
        for contour in self.fast_contours:
            if contour.match == None:
                return False
        return True
    
    def existing_contour_match(self, contour):
        for i, existing_contour in enumerate(self.matched_well_contours):
            if existing_contour != None and contour.overlap(existing_contour):
                return i, self.fast_contours[i]
        return None, None
    
    def delete_unmatched_contours(self):
        for contour_index in range(len(self.fast_contours) -1, -1, -1):
            if self.fast_contours[contour_index].match == None or self.matched_well_contours[contour_index] == None:
                del self.fast_contours[contour_index]
                del self.matched_well_contours[contour_index]
    
    def match_contours(self):
        iterations = 0
        while not self.all_contours_matched():
            if iterations >= 5:
                self.delete_unmatched_contours()
                break
            for contour_index, fast_contour in enumerate(self.fast_contours):
                for threshold_index, _ in enumerate(self.well_thresholds):
                    for well_contour in self.well_images[threshold_index].contours:
                        try:
                            well_contour = Contour(well_contour, threshold_index)
                            score = fast_contour.get_matching_score(well_contour)
                            if score < fast_contour.score:
                                if score >= 200:
                                    continue
                                match_index, match = self.existing_contour_match(well_contour)
                                if match != None:
                                    if score < match.score:
                                        match.unmatch_contour()
                                        self.matched_well_contours[match_index] = None
                                        fast_contour.match_contour(well_contour, score)
                                        self.matched_well_contours[contour_index] = well_contour
                                    else:
                                        continue
                                else:
                                    fast_contour.match_contour(well_contour, score)
                                    self.matched_well_contours[contour_index] = well_contour
                        except ContourTooSmallException as e:
                            continue
            iterations += 1
        return list(zip(self.matched_well_contours, self.fast_contours))

    def get_point_descriptors(self, points, image):
        sift = cv2.SIFT_create()
        keypoints = [cv2.KeyPoint(x=float(p[0]), y=float(p[1]), size=20) for p in points]
        keypoints, descriptors = sift.compute(image, keypoints)
        return descriptors

    def match_points(self, well_contour, fast_contour):
        well_points = well_contour.get_points()
        fast_points = fast_contour.get_points()
        well_descriptors = self.get_point_descriptors(well_points, self.well_images[well_contour.well_image_threshold_index].thresholded)
        fast_descriptors = self.get_point_descriptors(fast_points, self.fast_image.thresholded)
        bf = cv2.BFMatcher()
        matches = bf.match(fast_descriptors, well_descriptors)
        matches = sorted(matches, key=lambda x: x.distance)
        fast_matched_points = [fast_points[m.queryIdx] for m in matches]
        well_matched_points = [well_points[m.trainIdx] for m in matches]
        return well_matched_points, fast_matched_points

    def alignImages(self):
        matched_contours = self.match_contours()
        well_points_combined = []
        fast_points_combined = []
        for well_contour, fast_contour in matched_contours:
            well_matched_points, fast_matched_points = self.match_points(well_contour, fast_contour)
            well_points_combined.extend(well_matched_points)
            fast_points_combined.extend(fast_matched_points)
        if len(well_points_combined) >= 4:
            well_points = np.float32(well_points_combined)
            fast_points = np.float32(fast_points_combined)
            H, mask = cv2.findHomography(fast_points, well_points, cv2.RANSAC, 5.0)
            warped_image = cv2.warpPerspective(self.fast_image.image, H, (self.image_width, self.image_height))
        else:
            print("Not enough points for homography.")
        return warped_image, H

def save_transformation(json_path, right_filename, matrix):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {'matrices': []}

    found = False
    for item in data['matrices']:
        if item['name'] == right_filename:
            item['matrix'] = matrix.tolist()
            found = True
            break
    
    if not found:
        named_matrix = {'name': right_filename, 'matrix': matrix.tolist()}
        data['matrices'].append(named_matrix)

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

def create_mask(matrix):
    matrix = np.array(matrix, dtype=np.float32)
    h, w = 960, 960
    if np.all(matrix == 0):
        return np.ones((h, w), dtype=np.bool_)
    corners = np.array([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]], dtype=np.float32).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners, matrix)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(transformed_corners), 1)
    return mask

def delete_unevaluated_images_from_json(transfoms_json_path, eval_images):
    with open(transfoms_json_path, 'r') as json_file:
        data = json.load(json_file)

    frames_to_keep = []
    for frame in data.get('frames', []):
        file_path = frame.get('file_path', '')
        file_name = os.path.basename(file_path)
        if file_name in eval_images:
            frames_to_keep.append(frame)
        elif 'left_e1' in file_name and file_name.replace('left_e1', 'right_e2') in eval_images:
            frames_to_keep.append(frame)
    data['frames'] = frames_to_keep

    with open(transfoms_json_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def change_code_to_eval_all_images(all_images):
    with open('nerfstudio/data/datamanagers/base_datamanager.py', 'r') as file:
        content = file.readlines()

    new_content = []
    for line in content:
        if 'to eval all images' in line:
            if all_images == True:
                new_content.append(line.replace('# self.eval_dataset', 'self.eval_dataset'))
            else:
                new_content.append(line.replace('self.eval_dataset', '# self.eval_dataset'))
        else:
            new_content.append(line)

    with open('nerfstudio/data/datamanagers/base_datamanager.py', 'w') as file:
        file.writelines(new_content)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input_dir", type=str, default="/mnt/data/scene/scene_ns/", help="Nerfstudio directory of the scene")
    argparser.add_argument("--config", type=str, default="outputs/scene_ns/lantern-nerfacto/2024-06-20_142413/config.yml", help="Nerfstudio config file")
    argparser.add_argument("--no_eval", action='store_false', default="True", help="Doesn't run eval if flag is present")
    argparser.add_argument("--no_copy", action='store_false', default="True", help="Doesn't copy the original data if flag is present. Use if already copied.")
    args = argparser.parse_args()

    if args.no_eval:
        print("Running eval...")
        change_code_to_eval_all_images(True)
        subprocess.run(["ns-eval", "--load-config=" + args.config, "--output-path=output.json",  "--render_output_path=" + args.input_dir + "eval_for_alignment"])
        change_code_to_eval_all_images(False)

    eval_path = os.path.join(args.input_dir, 'eval_for_alignment')
    json_path = os.path.join(args.input_dir, 'alignment_matrices.json')
    transfoms_json_path = os.path.join(args.input_dir, 'transforms.json')
    images_path_aligned = os.path.join(args.input_dir, 'images')
    images_path_unaligned = images_path_aligned + '_unaligned'
    masks_path_aligned = os.path.join(args.input_dir, 'masks')
    masks_path_unaligned = masks_path_aligned + '_unaligned'
    merged_images = os.path.join(images_path_aligned, 'merged')
    
    if args.no_copy:
        print("Copying data...")
        subprocess.run(["cp", "-r", images_path_aligned, images_path_unaligned])
        os.makedirs(merged_images, exist_ok=True)
        subprocess.run(["cp", "-r", masks_path_aligned, masks_path_unaligned])
        subprocess.run(["cp", "-r", transfoms_json_path, os.path.join(args.input_dir, 'transforms_unaligned.json')])
    os.makedirs(merged_images, exist_ok=True)

    print("Aligning images...")
    eval_images = []
    for right_name in os.listdir(images_path_unaligned):
        if right_name.startswith("right") and right_name.endswith(".png"):
            left_eval_image_path = os.path.join(eval_path, right_name.replace('.png', '-img.png'))
            if os.path.exists(left_eval_image_path):
                eval_images.append(right_name)
                left_image = cv2.cvtColor(cv2.imread(left_eval_image_path), cv2.COLOR_BGR2RGB)
                right_path = os.path.join(images_path_unaligned, right_name)
                right_image = cv2.cvtColor(cv2.imread(right_path), cv2.COLOR_BGR2RGB)

                try:
                    align_images = AlignImages(left_image, right_image)
                    warped_image, matrix = align_images.alignImages()
                    print(f"Aligned: {right_name}")
                except Exception as e:
                    warped_image = right_image
                    matrix = np.zeros((3, 3))
                    print(f"Failed to align: {right_name}")

                cv2.imwrite(os.path.join(images_path_aligned, right_name), cv2.cvtColor(warped_image, cv2.COLOR_RGB2BGR))
                save_transformation(json_path, right_name, matrix)

                mask = create_mask(matrix)
                mask_path = os.path.join(masks_path_unaligned, right_name)
                previous_mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2GRAY)
                merged_masks = (1 - np.logical_or(1 - mask.astype(np.bool_), 1 - previous_mask.astype(np.bool_))).astype(np.uint8) * 255
                cv2.imwrite(os.path.join(masks_path_aligned, right_name), merged_masks)

                alpha = 50.0/100.0
                merged_image = cv2.addWeighted(left_image, 1.0-alpha, warped_image, alpha, 0)
                merged_masked_image = cv2.bitwise_or(merged_image, merged_image, mask=merged_masks)
                merged_path = os.path.join(merged_images, right_name + '.merged.png')
                cv2.imwrite(merged_path, cv2.cvtColor(merged_masked_image, cv2.COLOR_RGB2BGR))
    
    print("Modifying json...")
    delete_unevaluated_images_from_json(transfoms_json_path, eval_images)