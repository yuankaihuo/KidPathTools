import openslide
import xmltodict
import numpy as np
from PIL import Image
import os
import cv2
from read_mask_Zheyu import read_mask
import glob



if __name__ == "__main__":
    match_file = True

    if match_file:
        # source_dir = '/home/zheyuzhu/Desktop/022619/0.2'
        # output_dir = '/home/zheyuzhu/Desktop/022619/0.2_good_bad'
        # Good_Bad_dir = '/media/zheyuzhu/My Passport/qa/022619/0.2qa'

        source_dir = '/home/zheyuzhu/Desktop/081617_ADE/0.2'
        output_dir = '/home/zheyuzhu/Desktop/081617_ADE/0.2_good_bad'
        Good_Bad_dir = '/media/zheyuzhu/My Passport/qa/081617_ADE/0.2qa'


        xml_files = glob.glob(os.path.join(source_dir,'*.xml'))
        for i in range(len(xml_files)):
            xml_file = xml_files[i]
            simg= None
            read_mask(simg,  xml_file, output_dir, Good_Bad_dir, match_file=match_file)

    else:
        source_dir = '/media/zheyuzhu/TOSHIBA EXT/newdata/R24_Project/17-301_2017-11-21 12_55_39'
        output_dir = '/media/zheyuzhu/TOSHIBA EXT/scn_image/R24_Project/0.2/17-301_2017-11-21 12_55_39'


        scn_files1 = glob.glob(os.path.join(source_dir,'*.scn'))
        scn_files2 = glob.glob(os.path.join(source_dir, '*.sys'))
        scn_files = scn_files1 + scn_files2
        scn_files.sort()

        for i in range(len(scn_files)):
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
            read_mask(simg, xml_file, output_dir ,match_file=match_file)



