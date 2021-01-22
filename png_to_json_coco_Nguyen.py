import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools


INPUT_ROOT_DIR = '/home/sybbure/CircleNet/MoNuSeg-Test/coco-mask'

ROOT_DIR = '/home/sybbure/CircleNet/MoNuSeg-Test/MoNuSeg-coco'

INFO = {
    "description": "MICCAI 2018 - MoNuSeg",
    "url": "https://monuseg.grand-challenge.org/Home/",
    "version": "0.1.0",
    "year": 2018,
    "contributor": "EthanHNguyen",
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
        'name': 'nuclei',
        'supercategory': 'nuclei',
    }
]


def filter_for_jpeg(root, files):
    file_types = ['*.jpeg', '*.jpg']
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

def walklevel(some_dir, level=1):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)

    num_sep = some_dir.count(os.path.sep)

    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]

def main():
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1

    # filter for jpeg images
    for root, folders, files in walklevel(INPUT_ROOT_DIR, 0):
        # image_files = filter_for_jpeg(root, files)

        # go through each image
        for image_folder in folders:
            # print(image_folder)
            folder_path = os.path.join(INPUT_ROOT_DIR, image_folder)
            image_path = os.path.join(folder_path, 'image', 'image.png')

            image = Image.open(image_path)
            image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_folder), image.size)

            print(image_info)

            coco_output["images"].append(image_info)

            INPUT_MASK_DIR = os.path.join(folder_path, 'masks')

            # filter for associated png annotations
            for root, _, files in os.walk(INPUT_MASK_DIR):
                annotation_files = files

                # go through each associated annotation
                for annotation_filename in annotation_files:
                    annotation_path = os.path.join(folder_path, 'masks', annotation_filename)
                    IMAGE_DIR = os.path.join(ROOT_DIR, 'train')

                    # Create directories if they don't exist
                    if not os.path.exists(IMAGE_DIR):
                        os.makedirs(IMAGE_DIR)

                    image_filename_new = os.path.join(ROOT_DIR, 'train', image_folder + '.png')
                    os.system("cp '%s' '%s'" % (image_path, image_filename_new))

                    # class_id = [x['id'] for x in CATEGORIES if x['name'] in new_annotation_name][0]
                    class_id = 1

                    category_info = {'id': class_id, 'is_crowd': 'crowd' in image_folder}
                    binary_mask = np.asarray(Image.open(annotation_path)
                                             .convert('1')).astype(np.uint8)

                    annotation_info = pycococreatortools.create_annotation_info(
                        segmentation_id, image_id, category_info, binary_mask,
                        image.size, tolerance=2)

                    # print(annotation_info)

                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)

                    segmentation_id = segmentation_id + 1

            image_id = image_id + 1

    with open(os.path.join(ROOT_DIR, 'MoNuSeg2018-train.json').format(ROOT_DIR), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)


if __name__ == "__main__":
    main()