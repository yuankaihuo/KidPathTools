import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools


INPUT_ROOT_DIR = '/home/sybbure/CircleNet/MoNuSeg-Test/png-instance-2/roi'

ROOT_DIR = '/home/sybbure/CircleNet/MoNuSeg-Test/png-coco-like-3'

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

def filter_for_xml(files):
    file_types = ['*.xml']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [f for f in files if re.match(file_types, f)]

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


    image_id = 1
    segmentation_id = 1

    sublist = {}
    sublist['train'] = ['TCGA-HE-7129-01Z-00-DX1', 'TCGA-A7-A13F-01Z-00-DX1', 'TCGA-KB-A93J-01A-01-TS1', 'TCGA-G9-6356-01Z-00-DX1', 'TCGA-B0-5711-01Z-00-DX1', 'TCGA-G9-6336-01Z-00-DX1', 'TCGA-CH-5767-01Z-00-DX1', 'TCGA-A7-A13E-01Z-00-DX1', 'TCGA-AR-A1AS-01Z-00-DX1', 'TCGA-G2-A2EK-01A-02-TSB', 'TCGA-G9-6348-01Z-00-DX1', 'TCGA-RD-A8N9-01A-01-TS1', 'TCGA-50-5931-01Z-00-DX1', 'TCGA-NH-A8F7-01A-01-TS1', 'TCGA-B0-5698-01Z-00-DX1', 'TCGA-HE-7128-01Z-00-DX1', 'TCGA-B0-5710-01Z-00-DX1', 'TCGA-AR-A1AK-01Z-00-DX1', 'TCGA-21-5784-01Z-00-DX1', 'TCGA-HE-7130-01Z-00-DX1']
    sublist['val'] = ['TCGA-18-5592-01Z-00-DX1', 'TCGA-38-6178-01Z-00-DX1', 'TCGA-21-5786-01Z-00-DX1', 'TCGA-G9-6363-01Z-00-DX1', 'TCGA-DK-A2I6-01A-01-TS1', 'TCGA-E2-A1B5-01Z-00-DX1', 'TCGA-G9-6362-01Z-00-DX1', 'TCGA-AY-A8YK-01A-01-TS1', 'TCGA-49-4488-01Z-00-DX1', 'TCGA-E2-A14V-01Z-00-DX1']
    sublist['test'] = ['TCGA-AC-A2FO-01A-01-TS1', 'TCGA-ZF-A9R5-01A-01-TS1', 'TCGA-EJ-A46H-01A-03-TSC', 'TCGA-CU-A0YN-01A-02-BSB', 'TCGA-HC-7209-01A-01-TS1', 'TCGA-FG-A4MU-01B-01-TS1', 'TCGA-A6-6782-01A-01-BS1', 'TCGA-HT-8564-01Z-00-DX1', 'TCGA-44-2665-01B-06-BS6', 'TCGA-IZ-8196-01A-01-BS1', 'TCGA-2Z-A9J9-01A-01-TS1', 'TCGA-69-7764-01A-01-TS1', 'TCGA-AO-A0J2-01A-01-BSA', 'TCGA-GL-6846-01A-01-BS1']
    types = ['train','val','test']

    # # Get files names for split
    # for root, folders, files in walklevel('/home/sybbure/CircleNet/MoNuSegTestData', 0):
    #     files = filter_for_xml(files)
    #     print(files)
    #     print(len(files))
    #     idx = 0
    #     for file in files:
    #         files[idx] = file.split('.')[0]
    #         idx += 1
    #     print(files[:20])
    #     print(len(files[:20]))
    #
    #     print(files[20:])
    #     print(len(files[20:]))

    for type in types:

        coco_output = {
            "info": INFO,
            "licenses": LICENSES,
            "categories": CATEGORIES,
            "images": [],
            "annotations": []
        }

        # filter for jpeg images
        for root, folders, files in walklevel(INPUT_ROOT_DIR, 0):
            # image_files = filter_for_jpeg(root, files)
            json_file = os.path.join(ROOT_DIR, 'MoNuSeg_%s2021.json' % type)

            # go through each image
            for image_folder in folders:
                # print(image_folder)
                folder_path = os.path.join(INPUT_ROOT_DIR, image_folder)
                image_path = os.path.join(folder_path, 'image', os.path.basename(folder_path)+'.png')

                image = Image.open(image_path)
                image_info = pycococreatortools.create_image_info(
                    image_id, os.path.basename(image_folder) + '.png', image.size)

                print(image_info)
                subname = os.path.basename(folder_path).split('-')[:6]
                subname = '-'.join(subname)
                if subname in sublist[type]:
                    coco_output["images"].append(image_info)

                    INPUT_MASK_DIR = os.path.join(folder_path, 'masks')

                    # filter for associated png annotations
                    for root, _, files in os.walk(INPUT_MASK_DIR):
                        annotation_files = files

                        # go through each associated annotation
                        for count, annotation_filename in enumerate(annotation_files):
                            if count % 100 == 0:
                                print('.', end='')
                            annotation_path = os.path.join(folder_path, 'masks', annotation_filename)
                            IMAGE_DIR = os.path.join(ROOT_DIR, type)

                            # Create directories if they don't exist
                            if not os.path.exists(IMAGE_DIR):
                                os.makedirs(IMAGE_DIR)

                            image_filename_new = os.path.join(ROOT_DIR, type, image_folder + '.png')
                            os.system("cp '%s' '%s'" % (image_path, image_filename_new))

                            # class_id = [x['id'] for x in CATEGORIES if x['name'] in new_annotation_name][0]
                            class_id = 1

                            category_info = {'id': class_id, 'is_crowd': 'crowd' in image_folder}
                            binary_mask = np.asarray(Image.open(annotation_path)
                                                     .convert('1')).astype(np.uint8)

                            annotation_info = pycococreatortools.create_annotation_info(
                                segmentation_id, image_id, category_info, binary_mask,
                                image.size, tolerance=2)

                            if annotation_info is not None:
                                coco_output["annotations"].append(annotation_info)

                            segmentation_id = segmentation_id + 1

                    image_id = image_id + 1
                    print()

        with open(json_file.format(ROOT_DIR), 'w') as output_json_file:
            json.dump(coco_output, output_json_file)


if __name__ == "__main__":
    main()