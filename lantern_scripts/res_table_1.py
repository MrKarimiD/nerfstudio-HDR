from ezexr import imread
import os
import numpy as np
from tqdm import tqdm
from math import log10
import json
import cv2
import imageio


def wrmse(gt, est, mask):
    if mask is None:
        gt = gt.flatten()
        est = est.flatten()
    else:
        gt = gt[mask].flatten()
        est = est[mask].flatten()
    error = np.sqrt(np.mean(np.power(gt - est, 2)))

    return error

def si_wrmse(gt, est, mask):
    if mask is None:
        gt_c = gt.flatten()
        est_c = est.flatten()
    else:
        gt_c = gt[mask].flatten()
        est_c = est[mask].flatten()
    alpha = (np.dot(np.transpose(gt_c), est_c)) / (np.dot(np.transpose(est_c), est_c))
    error = wrmse(gt, est * alpha, mask)

    return error

def tonemap(img, gamma=2.4):
    """Apply gamma, then clip between 0 and 1, finally convert to uint8 [0,255]"""
    return (np.clip(np.power(img,1/gamma), 0.0, 1.0)*255).astype('uint8')


def angular_error(gt_render, pred_render, mask=None):
    # The error need to be computed with the normalized rgb image.
    # Normalized RGB is r = R / (R+G+B), g = G / (R+G+B), b = B / (R+G+B)
    # The angular distance is the distance between pixel 1 and pixel 2.
    # It's computed with cos^-1(p1·p2 / ||p1||*||p2||)
    gt_norm = np.empty((gt_render.shape))
    pred_norm = np.empty(pred_render.shape)

    for i in range(3):
        gt_norm[:,:,i] = gt_render[:,:,i] / np.sum(gt_render, axis=2, keepdims=True)[:,:,0]
        pred_norm[:,:,i] = pred_render[:,:,i] / (np.sum(pred_render, axis=2, keepdims=True)[:,:,0] + 1e-8)

    angular_error_arr = np.arccos( np.sum(gt_norm*pred_norm, axis=2, keepdims=True)[:,:,0] / 
        ((np.sqrt(np.sum(gt_norm*gt_norm, axis=2, keepdims=True)[:,:,0])*np.sqrt(np.sum(pred_norm*pred_norm, axis=2, keepdims=True)[:,:,0]))) )

    if mask is not None:
        angular_error_arr = angular_error_arr[mask[:,:,0]]
    else:
        angular_error_arr = angular_error_arr.flatten()
    angular_error_arr = angular_error_arr[~np.isnan(angular_error_arr)]
    mean = np.mean(angular_error_arr)
    # convert to degree
    mean = mean * 180 / np.pi
    return mean


def psnr(original, compressed):
    original_tone_mapped = tonemap(original)
    compressed_tone_mapped = tonemap(compressed)

    cv2.imwrite('./out.png',  cv2.cvtColor(compressed_tone_mapped, cv2.COLOR_RGB2BGR))    
    cv2.imwrite('./gt.png',  cv2.cvtColor(original_tone_mapped, cv2.COLOR_RGB2BGR))  

    mse = wrmse(original_tone_mapped, compressed_tone_mapped, None)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255 #max(original.max(), compressed.max())
    psnr = 20 * log10(max_pixel / mse)
    return psnr


def remove_nan_and_sort(x):
    x = np.array(x)
    x = x[~np.isnan(x)]
    x = np.sort(x)
    return x


results_dataset_roots = [ "/mnt/data/metrics/renders/lab_downstairs_lin_renders/"
                        # '/Users/momo/Desktop/Pandora-HDR_NeRF/final_results/meeting_room/LDR-Nerfacto/render/',
                        # '/Users/momo/Desktop/Pandora-HDR_NeRF/final_results/meeting_room/linearization_then_nerf/render/',
                        # '/Users/momo/Desktop/Pandora-HDR_NeRF/final_results/meeting_room/nerf_then_linearization/render/',
                        # '/Users/momo/Desktop/Pandora-HDR_NeRF/final_results/meeting_room/HDR-Nerfacto/render/',
                        # '/Users/momo/Desktop/Pandora-HDR_NeRF/final_results/meeting_room/PanoHDR-NeRF/render/'
                        ]
# results_dataset_roots = '/home/karimida/evaluation/final_results/Henrique_data/my_renderings_of_Henrique_2'

dataset_names = [ "lab_downstairs_lin"
                # 'LDR_nerfacto',
                # 'linearization_then_NeRF',
                # 'NeRF_then_linearization',
                # 'HDR-Nerfacto',
                # 'PanoHDRNeRF'
                ]

mse = {}
si_mse = {}
angular = {}
psnr_error = {}
for key in dataset_names:
    mse[key] = []
    si_mse[key] = []
    angular[key] = []
    psnr_error[key] = []

# Ground truth for the indoor
# gt_dataset_root = '/home/karimida/evaluation/redoing_table_one/hdr_env_maps/gt_render_diffuse/'

# Ground truth for the outdoor
# gt_dataset_root = '/Users/momo/Desktop/Pandora-HDR_NeRF/final_results/meeting_room/GT/render/'
gt_dataset_root = '/mnt/data/metrics/GT_lab_downstairs/render/'
# output_folder = '/Users/momo/Desktop'
output_folder = '/mnt/data/metrics/lab_downstairs/res_table/'

gt_dataset_files = sorted([os.path.join(gt_dataset_root, f) for f in os.listdir(gt_dataset_root) if f.endswith('.exr')])

mask_dataset_root = '/mnt/data/metrics/GT_lab_downstairs/render_mask/'

# create output folder output_folder, 'metrics'
os.system('mkdir -p ' + output_folder + '/metrics')

for results_dataset_root, dataset_name in zip(results_dataset_roots, dataset_names):
    
    for gt_dataset_file in tqdm(gt_dataset_files):
        gt_dataset_img_exr = imread(gt_dataset_file)[:, :, :3].astype(np.float32)
    
        result_dataset_file = gt_dataset_file.replace(gt_dataset_root, results_dataset_root)
        result_dataset_img_exr = imread(result_dataset_file)[:, :, :3].astype(np.float32)

        mask_file = gt_dataset_file.replace(gt_dataset_root, mask_dataset_root)
        mask_file = mask_file.replace('.exr', '.png')
        mask = np.float32(imageio.imread(mask_file)) / 255.0
        mask = mask == 1
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        gt_dataset_img_exr_for_PSNR = gt_dataset_img_exr[mask]
        result_dataset_img_exr_for_PSNR = result_dataset_img_exr[mask]
        
        mse_result = wrmse(gt_dataset_img_exr, result_dataset_img_exr, mask) 
        scale_invariant_mse_result = si_wrmse(gt_dataset_img_exr, result_dataset_img_exr, mask)
        angular_result = angular_error(gt_dataset_img_exr, result_dataset_img_exr, mask)
        psnr_result = psnr(gt_dataset_img_exr_for_PSNR, result_dataset_img_exr_for_PSNR)
        mse[dataset_name].append(mse_result)
        si_mse[dataset_name].append(scale_invariant_mse_result)
        angular[dataset_name].append(angular_result)
        psnr_error[dataset_name].append(psnr_result)
        si_mse[dataset_name].append(scale_invariant_mse_result)



    # mse = {key: remove_nan_and_sort(mse[key]) for key in mse}
    mse[dataset_name] = remove_nan_and_sort(mse[dataset_name]) 

    # mse_ours = np.percentile(mse[dataset_name], q=50)
    # si_mse_ours = np.percentile(si_mse[dataset_name], q=50)
    # angular_ours = np.percentile(angular[dataset_name], q=50)
    # psnr_ours = np.percentile(psnr_error[dataset_name], q=50)

    mse_ours = np.mean(mse[dataset_name])
    si_mse_ours = np.mean(si_mse[dataset_name])
    angular_ours = np.mean(angular[dataset_name])
    psnr_ours = np.mean(psnr_error[dataset_name])

    # print(dataset_name + ', si_RMSE: {}'.format(si_mse_ours))
    # print(dataset_name + ', RMSE: {}'.format(mse_ours))
    # print(dataset_name + ', angular: {}'.format(angular_ours))
    # print(dataset_name + ', PSNR: {}'.format(psnr_ours))

    output_dict = {}
    output_dict['model_name'] = dataset_name
    output_dict['si_RMSE'] = str(si_mse_ours)
    output_dict['RMSE'] = str(mse_ours)
    output_dict['angular'] = str(angular_ours)
    output_dict['PSNR'] = str(psnr_ours)

    output_addr = output_folder + '/metrics/' + dataset_name + '_results.json'
    # print("output_addr: ", output_addr)
    with open(output_addr, 'w') as f:
        json.dump(output_dict, f, indent=4)