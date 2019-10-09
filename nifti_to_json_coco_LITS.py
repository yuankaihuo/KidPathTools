import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
import glob
import pynifticreatortools
from skimage.measure import label
import nibabel as nib
previous_labels = []
previous_labels.append('/home/yuankai/Projects/CT_liver/CVPR/data/cgmh200/cgmh200_train2019.json')
previous_labels.append('/home/yuankai/Projects/CT_liver/CVPR/data/cgmh200/cgmh200_val2019_true.json')


INPUT_ROOT_DIR = '/home/yuankai/Projects/CT_liver/CVPR/preprocessing/Crop_48_256_176/data_LITS_131subs/'
MASK_ROOT_DIR = '/home/yuankai/Projects/CT_liver/CVPR/preprocessing/Crop_48_256_176/mask_LITS_131subs/'

ROOT_DIR = '/home/yuankai/Projects/CT_liver/CVPR/data/lits'


INFO = {
    "description": "lits Dataset",
    "url": "http",
    "version": "0.1.0",
    "year": 2019,
    "contributor": "Yuankai Huo",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'lesion',
        'supercategory': 'lesion',
    }
]



def filter_for_nifti(root, files):
    file_types = ['*.nii.gz']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    return files



def filter_for_jpeg(root, files):
    file_types = ['*.jpeg', '*.jpg', '*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    return files


def filter_for_annotations(root, files, image_filename):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]
    return files

def if_in_dict(data_list, subname):
    for data in data_list:
        if data['name'] == subname:
            return True
    return False

def if_in_list(data_list, subname):
    for data in data_list:
        if data == subname:
            return True
    return False



def filedict_from_json(json_path, key):
    """
    Functions to create lists of filenames and/or labels.
    Should return a list of dictionaries, where each
    dictionary specifies a data instance.

    E.g.,
        [{'image': path, 'label': path}...]
    """

    with open(json_path, 'r') as f:
        json_dict = json.load(f)

    listdict = json_dict.get(key, [])

    return listdict

def get_largest_image_segmentation_id(json_file):
    dataset = json.load(open(json_file, 'r'))
    image_id_max = -1
    for ii in range(len(dataset['images'])):
        id = dataset['images'][ii]['id']
        image_id_max = max(image_id_max,id)

    segmentation_id_max = -1
    for si in range(len(dataset['images'])):
        sid = dataset['annotations'][si]['id']
        segmentation_id_max = max(segmentation_id_max,sid)

    return image_id_max, segmentation_id_max


def main():

    #get larget image_id and segmentation_id
    image_id_max = 0
    segmentation_id_max = 0
    for ai in range(len(previous_labels)):
        image_id, segmentation_id =  get_largest_image_segmentation_id(previous_labels[ai])
        image_id_max = max(image_id_max, image_id)
        segmentation_id_max = max(segmentation_id_max, segmentation_id)

    if image_id_max == 0 or segmentation_id_max == 0:
        print('something is wrong')
        return

    image_id = image_id_max + 1
    segmentation_id = segmentation_id_max + 1
    fold = 0

    sublist = {}
    sublist['train'] = []
    sublist['val'] = []
    for i in range(0, 101):
        sublist['train'].append('liver_%d'%i)
    for i in range(101, 131):
        sublist['val'].append('liver_%d'%i)

    phases = ['venous_phase']
    types = ['train','val']


    # #rename labels
    # labels = glob.glob('/home/yuankai/Projects/CT_liver/detection/datasets/kidpath/annotations/*.png')
    # for li in range(len(labels)):
    #     label = labels[li]
    #     label_new = label.replace('.png','_glomerulus_0.png')
    #     os.system('mv "%s" "%s"'%(label,label_new))

    for type in types:

        coco_output = {
            "info": INFO,
            "licenses": LICENSES,
            "categories": CATEGORIES,
            "images": [],
            "annotations": []
        }
        # filter for jpeg images
        root = INPUT_ROOT_DIR
        # files = glob.glob(os.path.join(INPUT_ROOT_DIR,'*','venous_phase.nii.gz'))
        # files.sort()

        # image_files = filter_for_nifti(root, files)
        json_files = os.path.join(ROOT_DIR,'lits_%s2019.json'%type)
        # go through each image
        for ii in range(len(sublist[type])):
            subname = sublist[type][ii]
            image_filename = os.path.join(INPUT_ROOT_DIR, subname, 'venous_phase.nii.gz')
            image = nib.load(image_filename)
            # subname = os.path.basename(os.path.split(image_filename)[0])
            image_info = pynifticreatortools.create_image_info(
                image_id, subname, phases, image.shape)


            if not if_in_list(sublist[type], subname):
                continue
            else:
                coco_output["images"].append(image_info)
                # filter for associated png annotations

                IMAGE_DIR = os.path.join(ROOT_DIR, type)
                ANNOTATION_DIR = os.path.join(ROOT_DIR, 'mask_'+type)
                if not os.path.exists(IMAGE_DIR):
                    os.makedirs(IMAGE_DIR)
                if not os.path.exists(ANNOTATION_DIR):
                    os.makedirs(ANNOTATION_DIR)

                for pi in range(len(phases)):
                    phase = phases[pi]
                    image_filename_old = os.path.join(INPUT_ROOT_DIR, subname, phase+'.nii.gz')
                    image_filename_new = os.path.join(IMAGE_DIR, '%s-x-%s.nii.gz'%(subname, phase))
                    annotation_filename_old = os.path.join(MASK_ROOT_DIR, subname, 'rois_all-labels_corrected.nii.gz')
                    annotation_filename_new = os.path.join(ANNOTATION_DIR, '%s-x-mask.nii.gz'%(subname))
                    os.system("cp '%s' '%s'" %(image_filename_old, image_filename_new))
                    os.system("cp '%s' '%s'" % (annotation_filename_old, annotation_filename_new))

                # if annotation_filename_new == '/home/yuankai/Projects/CT_liver/detection/datasets/kidpath_multiROI/mask_train/Case 01-1-x-contour001-x-3082-x-1250_glomerulus_0.png':
                #     aaa = 1

                print(annotation_filename_new + ', count=%d'%ii )
                # class_id = [x['id'] for x in CATEGORIES if x['name'] in annotation_filename_new][0]
                class_id = 1

                category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename_new}
                # binary_mask = np.asarray(Image.open(annotation_filename)
                #                          .convert('1')).astype(np.uint8)
                mask_3d = nib.load(annotation_filename_new)
                mask = mask_3d.get_data()
                mask[mask>0] = 1
                binary_mask = mask.astype(np.uint8)
                # Image.fromarray(binary_mask.astype(np.uint8) * 100).show()

                labels = label(binary_mask)
                bincounts = np.bincount(labels.flat)
                sorted_ind = np.argsort(-bincounts)

                if len(bincounts) == 1:
                    continue

                max_size = max(bincounts[sorted_ind[1:]])

                for oi in range(1, len(sorted_ind)):
                    seg_id = sorted_ind[oi]
                    ROI_binary_mask = labels == seg_id
                    curr_size = ROI_binary_mask.sum()

                    if curr_size*10 < max_size and curr_size<300:
                        print('------ignore size = %d'% curr_size)
                        continue
                    else:
                        print('[keep size] = %d' % curr_size)

                    # Image.fromarray(ROI_binary_mask.astype(np.uint8) * 100).show()

                    # ROI_binary_mask = ROI_binary_mask[:,:,30]  #debug
                    annotation_info = pynifticreatortools.create_annotation_info(
                        segmentation_id, image_id, category_info, ROI_binary_mask,
                        image.shape, tolerance=2)



                    if annotation_info is None: # one pixel
                        continue
                    # if 0 in annotation_info['bbox'][0:3]: # touch top or left
                    #     continue
                    # if annotation_info['bbox'][0] + annotation_info['bbox'][3] == binary_mask.shape[1]: #touch right bordar:
                    #     continue
                    # if annotation_info['bbox'][1] + annotation_info['bbox'][4] == binary_mask.shape[0]:  # touch right bordar:
                    #     continue
                    # if annotation_info['bbox'][2] + annotation_info['bbox'][5] == binary_mask.shape[2]:  # touch down bordar:
                    #     print('touch down board')
                    #     continue


                    coco_output["annotations"].append(annotation_info)
                    segmentation_id = segmentation_id + 1

                image_id = image_id + 1

        with open(json_files, 'w') as output_json_file:
            json.dump(coco_output, output_json_file)


if __name__ == "__main__":
    main()
