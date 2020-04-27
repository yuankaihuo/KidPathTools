import openslide
import xmltodict
import numpy as np
from PIL import Image
import os
import cv2
from read_mask_Yuzhe import read_mask
import glob



if __name__ == "__main__":
    # scn_file = '/home-local/pathology/raw/Case 15-1.scn'
    # xml_file = '/home-local/pathology/raw/Case 15-1.xml'

    proc_type = 'classification'  # 'classification'  'segmentation'

    if proc_type == 'classification':
        # source_dir = '/media/huoy1/48EAE4F7EAE4E264/Projects/from_haichun/R24 scan slides'
        # xml_root_dir = '/media/huoy1/48EAE4F7EAE4E264/Projects/from_haichun/R24 scan slides'
        # output_dir = '/media/huoy1/48EAE4F7EAE4E264/Projects/detection/ROI_images/R24'

        # source_dir = '/media/huoy1/48EAE4F7EAE4E264/Projects/from_haichun/batch_1_data/scn'
        # xml_root_dir = '/media/huoy1/48EAE4F7EAE4E264/Projects/from_haichun/batch_1_data/scn'
        # output_dir = '/media/huoy1/48EAE4F7EAE4E264/Projects/detection/ROI_images/batch1'


        source_dir = '/media/huoy1/48EAE4F7EAE4E264/Projects/from_haichun/R24 scan slides'
        xml_root_dir = '/media/huoy1/48EAE4F7EAE4E264/Projects/from_haichun/R24_scan_slides_manual_QA'
        output_dir = '/media/huoy1/48EAE4F7EAE4E264/Projects/detection/ROI_images/R24_QA'

        boarder = 1.2  # two times larger

    if proc_type == 'segmentation':
        source_dir = '/media/huoy1/48EAE4F7EAE4E264/Projects/from_haichun/batch_1_data/scn'
        xml_root_dir = '/media/huoy1/48EAE4F7EAE4E264/Projects/from_haichun/batch_1_data/scn'
        output_dir = '/media/huoy1/48EAE4F7EAE4E264/Projects/detection/ROI_images/batch1_boarder2'

        # source_dir = '/media/huoy1/48EAE4F7EAE4E264/Projects/from_haichun/R24 scan slides'
        # xml_root_dir = '/media/huoy1/48EAE4F7EAE4E264/Projects/from_haichun/R24 scan slides'
        # output_dir = '/media/huoy1/48EAE4F7EAE4E264/Projects/detection/ROI_images/R24_boarder2'

        boarder = 2.0  # two times larger

    # source_dir = '/Volumes/Yuzhe_Disk/Pathology/R24 scan slides'
    # output_dir = '/Volumes/Yuzhe_Disk/Pathology/Output_new'

    scn_files1 = glob.glob(os.path.join(source_dir, '*.scn'))
    scn_files2 = glob.glob(os.path.join(source_dir, '*.sys'))
    scn_files = scn_files1 + scn_files2
    scn_files.sort()


    # boarder = 2.0  # two times larger

    for i in range(len(scn_files)):
        scn_file = scn_files[i]

        basename = os.path.basename(scn_file)
        fname, surfix = os.path.splitext(basename)
        xml_file = os.path.join(xml_root_dir,fname+'.xml')

        if not os.path.exists(xml_file):
            continue

        # if basename == 'Case 18-1.scn':
        #     aaa = 1
        # else:
        #     continue

        try:
            simg = openslide.open_slide(scn_file)
        except:
            print('can not read %s'%scn_file)
            continue
        # print(simg.dimensions)

        # output_sub_dir = os.path.join(output_dir, fname)
        # if not os.path.exists(output_sub_dir):
        #     os.makedirs(output_sub_dir)
        read_mask(simg, xml_file, output_dir, boarder)



