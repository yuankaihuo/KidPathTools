import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
import glob
import pycococreatortools
from skimage.measure import label

INPUT_ROOT_DIRs = []
INPUT_ROOT_DIRs.append('/media/huoy1/MyDrive/pathology/coco-like/train_roi')
INPUT_ROOT_DIRs.append('/media/huoy1/48EAE4F7EAE4E264/Projects/fromZheyu/coco-like-081617_ADE/train_roi')

MASK_ROOT_DIRs = []
MASK_ROOT_DIRs.append('/media/huoy1/MyDrive/pathology/coco-like/labels_roi')
MASK_ROOT_DIRs.append('/media/huoy1/48EAE4F7EAE4E264/Projects/fromZheyu/coco-like-081617_ADE/labels_roi')

ROOT_DIR = '/media/huoy1/48EAE4F7EAE4E264/Projects/detection/data/kidpath_first_batch_081617_ADE'


INFO = {
    "description": "KidneyPath Dataset with 081617_ADE",
    "url": "http",
    "version": "0.1.0",
    "year": 2020,
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
        'name': 'glomerulus',
        'supercategory': 'glomerulus',
    }
]






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


def main():


    image_id = 1
    segmentation_id = 1

    sublists = []
    sublist = {}
    sublist['train'] = ['Case 01','Case 02','Case 06','Case 08','Case 14','Case 15','Case 17','Case 19','Case 22','Case 23','Case 24','Case 25']
    sublist['val'] = ['Case 11', 'Case 12', 'Case 18', 'Case 20']
    sublist['test'] = ['Case 03', 'Case 05', 'Case 09', 'Case 16']
    sublists.append(sublist)

    sublist = {}
    sublist['train'] = ['all']
    sublist['val'] = ['']
    sublist['test'] = ['']
    sublists.append(sublist)

    # types = ['train','validation','test']
    types = ['train', 'val', 'test']


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

        json_files = os.path.join(ROOT_DIR, 'kidney_first_batch_081617_ADE_%s2019.json' % type)

        # filter for jpeg images
        for di in range(len(INPUT_ROOT_DIRs)):
            INPUT_ROOT_DIR = INPUT_ROOT_DIRs[di]
            MASK_ROOT_DIR = MASK_ROOT_DIRs[di]
            sublist = sublists[di]

            for root, _, files in os.walk(INPUT_ROOT_DIR):
                image_files = filter_for_jpeg(root, files)

                # go through each image
                for image_filename in image_files:
                    image = Image.open(image_filename)
                    image_info = pycococreatortools.create_image_info(
                        image_id, os.path.basename(image_filename), image.size)

                    subname = os.path.basename(image_filename).split('-x-')[0].split('-')[0]
                    if (subname in sublist[type]) or sublist[type]==['all']:
                        coco_output["images"].append(image_info)
                        # filter for associated png annotations
                        for root, _, files in os.walk(MASK_ROOT_DIR):
                            annotation_files = filter_for_annotations(root, files, image_filename)

                            # go through each associated annotation
                            for annotation_filename in annotation_files:

                                IMAGE_DIR = os.path.join(ROOT_DIR, type)
                                ANNOTATION_DIR = os.path.join(ROOT_DIR, 'mask_'+type)
                                if not os.path.exists(IMAGE_DIR):
                                    os.makedirs(IMAGE_DIR)
                                if not os.path.exists(ANNOTATION_DIR):
                                    os.makedirs(ANNOTATION_DIR)

                                image_filename_new = image_filename.replace(INPUT_ROOT_DIR,IMAGE_DIR)
                                annotation_filename_new = annotation_filename.replace(MASK_ROOT_DIR,ANNOTATION_DIR).replace('.png','_glomerulus_0.png')
                                os.system("cp '%s' '%s'" %(image_filename, image_filename_new))
                                os.system("cp '%s' '%s'" % (annotation_filename, annotation_filename_new))

                                # if annotation_filename_new == '/home/yuankai/Projects/CT_liver/detection/datasets/kidpath_multiROI/mask_train/Case 01-1-x-contour001-x-3082-x-1250_glomerulus_0.png':
                                #     aaa = 1

                                print(annotation_filename_new)
                                class_id = [x['id'] for x in CATEGORIES if x['name'] in annotation_filename_new][0]

                                category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
                                binary_mask = np.asarray(Image.open(annotation_filename)
                                                         .convert('1')).astype(np.uint8)
                                # Image.fromarray(binary_mask.astype(np.uint8) * 100).show()

                                labels = label(binary_mask)
                                bincounts = np.bincount(labels.flat)
                                sorted_ind = np.argsort(-bincounts)

                                for oi in range(1, len(sorted_ind)):
                                    seg_id = sorted_ind[oi]
                                    ROI_binary_mask = labels == seg_id
                                    # Image.fromarray(ROI_binary_mask.astype(np.uint8) * 100).show()

                                    annotation_info = pycococreatortools.create_annotation_info(
                                        segmentation_id, image_id, category_info, ROI_binary_mask,
                                        image.size, tolerance=2)



                                    if annotation_info is None: # one pixel
                                        continue
                                    if 0 in annotation_info['bbox'][0:2]: # touch top or left
                                        continue
                                    if annotation_info['bbox'][0]+annotation_info['bbox'][2] == binary_mask.shape[1]: #touch right bordar:
                                        continue
                                    if annotation_info['bbox'][1] + annotation_info['bbox'][3] == binary_mask.shape[0]:  # touch right bordar:
                                        continue


                                    coco_output["annotations"].append(annotation_info)
                                    segmentation_id = segmentation_id + 1

                        image_id = image_id + 1


        with open(json_files, 'w') as output_json_file:
            json.dump(coco_output, output_json_file)


if __name__ == "__main__":
    main()
