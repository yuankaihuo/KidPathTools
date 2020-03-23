import openslide
import xmltodict
import numpy as np
from PIL import Image
import os
import cv2
from scn_to_png import scn_to_png, scn_to_png_whole_slide, save_cropped_img_mask, scn_to_png_atubular, scn_to_png_whole_slide_atubular, save_cropped_img_mask_atubular
import glob



if __name__ == "__main__":
    source_dir = '/media/huoy1/48EAE4F7EAE4E264/Projects/from_haichun/atubular slides'
    xml_dir = '/media/huoy1/48EAE4F7EAE4E264/Projects/from_haichun/atubular_slides_detection'
    output_dir = '/media/huoy1/MyDrive/pathology/coco-like_atubular'
    ROI_type = 'multi_ROI' #single ROI, each path one ROI, multi_ROI, each patch mutiple ROI



    xml_files = glob.glob(os.path.join(xml_dir, '*.xml'))
    xml_files.sort()


    create_number_per_image = 10
    img_size = [512,512]
    pixel_size = 2  # 4 micron for human, 2 micron for mouse

    roi_counts = {}
    for i in range(9,len(xml_files)):
        xml_file = xml_files[i]
        basename = os.path.basename(xml_file)
        # print('%d working on %s' % (i,basename))
        fname, surfix = os.path.splitext(basename)
        scn_file = os.path.join(source_dir,fname+'.svs')

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
        # scn_to_png_atubular(simg, xml_file, output_dir, create_number_per_image, img_size, fname, pixel_size)

        #all ROIs all images
        pixel_size_threshold = 400
        if basename == '13-260.xml':
            read_patch = True
        else:
            read_patch = False
        big_img_file, big_mask_file, img, mask  = scn_to_png_whole_slide_atubular(simg, xml_file, output_dir, create_number_per_image, img_size, pixel_size, pixel_size_threshold, fname, pixel_size, read_patch)

        #get croped image and masks
        roi_count = save_cropped_img_mask_atubular(img, mask, output_dir, create_number_per_image, img_size, fname)
        roi_counts[basename] = roi_count
        print('%s = %d' % (basename, roi_count))

    print(roi_counts)





