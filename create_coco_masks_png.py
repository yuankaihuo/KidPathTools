import openslide
import xmltodict
import numpy as np
from PIL import Image
import os
import cv2
from scn_to_png import scn_to_png, scn_to_png_whole_slide, save_cropped_img_mask
import glob



if __name__ == "__main__":
    # scn_file = '/home-local/pathology/raw/Case 15-1.scn'
    # xml_file = '/home-local/pathology/raw/Case 15-1.xml'

    source_dir = '/media/huoy1/MyDrive/pathology/scn'
    output_dir = '/media/huoy1/MyDrive/pathology/coco-like/'
    ROI_type = 'multi_ROI' #single ROI, each path one ROI, multi_ROI, each patch mutiple ROI


    scn_files = glob.glob(os.path.join(source_dir, '*.scn'))
    # scn_files = glob.glob(os.path.join(source_dir,'*.svs'))
    scn_files.sort()

    create_number_per_image = 10
    img_size = [512,512]

    for i in range(len(scn_files)):
        scn_file = scn_files[i]
        basename = os.path.basename(scn_file)
        # print('%d working on %s' % (i,basename))
        fname, surfix = os.path.splitext(basename)
        xml_file = os.path.join(source_dir,fname+'.xml')

        try:
            simg = openslide.open_slide(scn_file)
        except:
            print('%s',basename)
            continue
        # print(simg.dimensions)



        # output_sub_dir = os.path.join(output_dir, fname)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # one ROI per image
        # scn_to_png(simg, xml_file, output_dir, create_number_per_image, img_size, fname)

        #all ROIs all images
        big_img_file, big_mask_file  = scn_to_png_whole_slide(simg, xml_file, output_dir, create_number_per_image, img_size, fname)

        #get croped image and masks
        save_cropped_img_mask(big_img_file, big_mask_file, output_dir, create_number_per_image, img_size, fname)





