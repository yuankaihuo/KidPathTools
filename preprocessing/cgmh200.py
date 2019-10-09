import torch
import numpy as np
from PIL import Image
from skimage.transform import resize
import os
import nibabel as nib
from glob import glob
import pandas as pd

def get_none_zero_range(liver_file):
    liver_3d = nib.load(liver_file)
    liver = liver_3d.get_data()
    # get bounding box
    non_zeros = liver.nonzero()
    liver_mask = liver>0
    x0 = non_zeros[0].min()
    x1 = non_zeros[0].max()
    y0 = non_zeros[1].min()
    y1 = non_zeros[1].max()
    z0 = non_zeros[2].min()
    z1 = non_zeros[2].max()
    return x0, x1, y0, y1, z0, z1, liver_mask

def crop_data(img_file, x0, x1, y0, y1, z0, z1, non_zeros):
    img_3d = nib.load(img_file)
    vol = img_3d.get_data()
    vol[non_zeros ==0] = 0
    vol_crop = vol[x0:x1, y0:y1, z0:z1]
    return vol_crop, img_3d

if __name__ == '__main__':

    Huo200_dir = '/media/yuankai/Data/project/Huo200/Data/step0_raw'

    resize_data_dir = '/home/yuankai/Projects/CT_liver/CVPR/preprocessing/Crop_48_256_176/data_deeds_2000subs'
    resize_mask_dir = '/home/yuankai/Projects/CT_liver/CVPR/preprocessing/Crop_48_256_176/mask_deeds_2000subs'

    raw_datasheet = '/media/yuankai/Data/project/Huo200/data_sheet.xlsx'

    scan_types = ['arterial_phase.nii.gz','delay_phase.nii.gz','non-contrast.nii.gz','venous_phase.nii.gz']
    resize_dim = [256, 176, 48]

    df = pd.read_excel(raw_datasheet)

    sub_list = df['Name']
    categories = df['Category']
    labels = df['label']

    for i in range(len(sub_list)):
        sub_name = sub_list[i]
        category = categories[i]
        label = labels[i]
        # print('working on %d/%d'%(i,len(sub_list)))
        sub_input_dir = os.path.join(Huo200_dir, category, sub_name)

        mask_file = os.path.join(sub_input_dir, 'rois_all-labels_corrected.nii.gz')
        if not os.path.exists(mask_file):
            print('error is happening')

        liver_file = os.path.join(sub_input_dir, 'ashwin_liver_plus_manual_lesion.nii.gz')
        x0, x1, y0, y1, z0, z1, non_zeros = get_none_zero_range(liver_file)

        # if (z1 - z0) < 12:
        #     print(sub_name)
        #     continue

        sub_output_dir = os.path.join(resize_data_dir,sub_name)
        if not os.path.exists(sub_output_dir):
            os.makedirs(sub_output_dir)

        sub_mask_output_dir = os.path.join(resize_mask_dir,sub_name)
        if not os.path.exists(sub_mask_output_dir):
            os.makedirs(sub_mask_output_dir)

        #get cropped file
        mask_crop, mask_3d = crop_data(mask_file, x0, x1, y0, y1, z0, z1, non_zeros)
        output_mask_file = os.path.join(sub_mask_output_dir, 'rois_all-labels_corrected.nii.gz')
        if not os.path.exists(output_mask_file):
            mask_resize = resize(mask_crop, resize_dim, order=0, anti_aliasing=False,
                                 preserve_range=True)  # resize to [32, 128, 128]
            # seg_img = nib.Nifti1Image(vol-resize,img_3d.affine,img_3d.header)
            seg_mask = nib.Nifti1Image(mask_resize, affine=mask_3d.affine)
            nib.save(seg_mask, output_mask_file)

        # print(sub_name, x1-x0, y1-y0, z1-z0)
        # continue


        for scan_type in scan_types:
            scan = os.path.join(sub_input_dir,scan_type)
            output_nii_file = os.path.join(sub_output_dir,scan_type)
            if os.path.exists(output_nii_file):
                continue



            if not os.path.exists(scan):
                print('%s is not found'%scan)
            else:
                # nimg2d = vol_crop[:,:,12]
                # im = Image.fromarray(nimg2d).convert('RGB')

                vol_crop, img_3d = crop_data(scan, x0, x1, y0, y1, z0, z1, non_zeros)
                np.asarray(vol_crop)
                vol_crop = vol_crop.astype(float)
                vol_crop[vol_crop < -1000] = -1000.0
                vol_crop[vol_crop > 1000] = 1000.0
                vol_resize = resize(vol_crop, resize_dim, anti_aliasing=True)  # resize to [32, 128, 128]
                # seg_img = nib.Nifti1Image(vol-resize,img_3d.affine,img_3d.header)
                seg_img = nib.Nifti1Image(vol_resize, affine=img_3d.affine)
                nib.save(seg_img, output_nii_file)
                print('x = %d, y = %d, z = %d'%(vol_crop.shape[0],vol_crop.shape[1],vol_crop.shape[2]))





