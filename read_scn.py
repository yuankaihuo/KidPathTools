import openslide
import xmltodict
import numpy as np
from PIL import Image
import os
import cv2
from read_mask import read_mask
import glob



if __name__ == "__main__":
    # scn_file = '/home-local/pathology/raw/Case 15-1.scn'
    # xml_file = '/home-local/pathology/raw/Case 15-1.xml'

    source_dir = '/home-local/pathology/scn/'
    output_dir = '/home-local/pathology/ROIs/'

    scn_files = glob.glob(os.path.join(source_dir,'*.scn'))
    scn_files.sort()

    for i in range(16,len(scn_files)):
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



        output_sub_dir = os.path.join(output_dir, fname)
        if not os.path.exists(output_sub_dir):
            os.makedirs(output_sub_dir)
        read_mask(simg, xml_file, output_sub_dir)





