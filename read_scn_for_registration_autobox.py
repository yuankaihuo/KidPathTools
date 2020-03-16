import openslide
import xmltodict
import numpy as np
from PIL import Image
import os
import cv2
from read_mask_for_registration import read_mask
from read_mask_for_registration import read_annotations
import glob

# this is to grab haichun's new annotation

if __name__ == "__main__":
    # scn_file = '/home-local/pathology/raw/Case 15-1.scn'
    # xml_file = '/home-local/pathology/raw/Case 15-1.xml'

    source_dir = '/media/huoy1/48EAE4F7EAE4E264/Projects/detection/registration/data/atubular_slides'
    annotation_dir = '/media/huoy1/48EAE4F7EAE4E264/Projects/detection/registration/data/atubular_slides_tracking'
    auto_detect_dir = '/media/huoy1/48EAE4F7EAE4E264/Projects/detection/test_demo/13-261'
    output_dir = '/media/huoy1/48EAE4F7EAE4E264/Projects/detection/registration/data/output_mask_auto'

    scn_files = glob.glob(os.path.join(source_dir,'*.svs'))
    scn_files.sort()

    for i in range(1,len(scn_files)):
        scn_file = scn_files[i]
        basename = os.path.basename(scn_file)
        fname, surfix = os.path.splitext(basename)
        xml_file = os.path.join(annotation_dir,fname+'.xml')
        auto_det_xml_file = os.path.join(auto_detect_dir,fname+'.xml')

        if not os.path.exists(xml_file):
            continue
        if not os.path.exists(auto_det_xml_file):
            continue

        try:
            simg = openslide.open_slide(scn_file)
        except:
            print('can not read %s'%scn_file)
            continue

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        read_annotations(simg, xml_file, output_dir, auto_det_xml_file)



