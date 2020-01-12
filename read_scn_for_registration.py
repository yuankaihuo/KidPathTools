import openslide
import xmltodict
import numpy as np
from PIL import Image
import os
import cv2
from read_mask_for_registration import read_mask
import glob



if __name__ == "__main__":
    # scn_file = '/home-local/pathology/raw/Case 15-1.scn'
    # xml_file = '/home-local/pathology/raw/Case 15-1.xml'

    source_dir = '/media/huoy1/48EAE4F7EAE4E264/Projects/detection/registration/data/atubular_slides'
    output_dir = '/media/huoy1/48EAE4F7EAE4E264/Projects/detection/registration/data/output_overlay'

    scn_files = glob.glob(os.path.join(source_dir,'*.svs'))
    scn_files.sort()

    for i in range(1,len(scn_files)):
        scn_file = scn_files[i]
        basename = os.path.basename(scn_file)
        fname, surfix = os.path.splitext(basename)
        xml_file = os.path.join(source_dir,fname+'.xml')

        try:
            simg = openslide.open_slide(scn_file)
        except:
            print('can not read %s'%scn_file)
            continue
        # print(simg.dimensions)

        # output_sub_dir = os.path.join(output_dir, fname)
        # if not os.path.exists(output_sub_dir):
        #     os.makedirs(output_sub_dir)
        read_mask(simg, xml_file, output_dir)



