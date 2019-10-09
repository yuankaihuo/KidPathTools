dataset = {}
dataset['cgmh200'] = {}
dataset['cgmh200']['train'] = {}
dataset['cgmh200']['val'] = {}
dataset['cgmh200']['train']['INPUT_ROOT_DIR'] = '/home/yuankai/Projects/CT_liver/CVPR/data/cgmh200/train'
dataset['cgmh200']['val']['INPUT_ROOT_DIR'] = '/home/yuankai/Projects/CT_liver/CVPR/data/cgmh200/val'
dataset['cgmh200']['train']['JSON_FILE'] = '/home/yuankai/Projects/CT_liver/CVPR/data/cgmh200/cgmh200_train2019.json'
dataset['cgmh200']['val']['JSON_FILE'] = '/home/yuankai/Projects/CT_liver/CVPR/data/cgmh200/cgmh200_val2019.json'

dataset['lits'] = {}
dataset['lits']['train'] = {}
dataset['lits']['val'] = {}
dataset['lits']['train']['INPUT_ROOT_DIR'] = '/home/yuankai/Projects/CT_liver/CVPR/data/lits/train'
dataset['lits']['val']['INPUT_ROOT_DIR'] = '/home/yuankai/Projects/CT_liver/CVPR/data/lits/val'
dataset['lits']['train']['JSON_FILE'] = '/home/yuankai/Projects/CT_liver/CVPR/data/lits/lits_train2019.json'
dataset['lits']['val']['JSON_FILE'] = '/home/yuankai/Projects/CT_liver/CVPR/data/lits/lits_val2019.json'

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


def main():
    types = ['train','val']
    merge_list = ['cgmh200','lits']

    for mi in range(len(merge_list)):
        if mi == 0:
            merge_name = merge_list[mi]
        else:
            merge_name = merge_name+'-x-'+merge_list[mi]

    OUTPUT_ROOT_DIR = os.path.join('/home/yuankai/Projects/CT_liver/CVPR/data',merge_name)


    for ti in range(len(types)):
        type = types[ti]
        output_image_dir = os.path.join(OUTPUT_ROOT_DIR,type)
        if not os.path.exists(output_image_dir):
            os.makedirs(output_image_dir)

        for mi in range(len(merge_list)):
            dataset_name = merge_list[mi]
            input_image_dir = dataset[dataset_name][type]['INPUT_ROOT_DIR']
            input_json_file = dataset[dataset_name][type]['JSON_FILE']
            json_dataset = json.load(open(input_json_file, 'r'))

            os.system("cp -r %s/* %s"%(input_image_dir,output_image_dir))
            if mi == 0:
                final_output = json_dataset
            else:
                for ii in range(len(json_dataset['images'])):
                    final_output['images'].append(json_dataset['images'][ii])
                for ai in range(len(json_dataset['annotations'])):
                    final_output['annotations'].append(json_dataset['annotations'][ii])

        final_output['info']['description'] = '%s Dataset'% merge_name
        final_output['info']['date_created'] = datetime.datetime.utcnow().isoformat(' ')

        json_files = os.path.join(OUTPUT_ROOT_DIR, '%s_%s2019.json' % (merge_name,type))

        with open(json_files, 'w') as output_json_file:
            json.dump(final_output, output_json_file)



if __name__ == "__main__":
    main()
