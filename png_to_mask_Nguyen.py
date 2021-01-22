import xmltodict
import os
import numpy as np
from PIL import Image
from skimage.measure import label
import cv2

def png_to_big_mask(png_file, xml_file, output_dir):
    # Read the XML file
    with open(xml_file) as fd:
        doc = xmltodict.parse(fd.read())
    layers = doc['Annotations']['Annotation']
    # contours = layers['Regions']['Region']
    # print(layers['Regions'])
    try:
        contours = layers['Regions']['Region']
    except:
        # Choose the layer with the mask
        Masklayer = layers[0]
        contours = Masklayer['Regions']['Region']

    # Create a unique directory for each image
    output_img_dir = os.path.join(output_dir, file.split(".")[0])

    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)

    img_dir = os.path.join(output_img_dir, "image")
    mask_dir = os.path.join(output_img_dir, "masks")

    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)

    # Open the image
    img = Image.open(png_file)
    img = np.array(img)  # read in image as NumPy array

    # For each region of interest, create a uniquely valued mask and save
    for ci in range(len(contours)):
        mask = np.zeros(img.shape, dtype=np.uint8)
        contour = contours[ci]
        vertices = contour["Vertices"]["Vertex"]

        # Read data into array
        # For each vertex in the ROI
        cnt = np.zeros((len(vertices), 1, 2))
        for vi in range(len(vertices)):
            # Read in x and y co-ordinate of each vertex
            cnt[vi, 0, 0] = float(vertices[vi]['@X'])
            cnt[vi, 0, 1] = float(vertices[vi]['@Y'])

        # Draw contours (note - color is green)
        # cv2.drawContours(cimg, [cnt.astype(int)], -1, (0, 255, 0), 1)

        # Draw masks (note - color is white and thickness = -1)
        cv2.drawContours(mask, [cnt.astype(int)], -1, (255, 255, 255), -1)

        mask_name = "mask_%s.png" % ci
        # cimg_out_file = os.path.join(output_cimg_dir, )
        mask_out_file = os.path.join(mask_dir, mask_name)

        # cimg_out = Image.fromarray(np.concatenate((img, cimg, mask), axis=1), 'I;16')
        # cimg_out.save(cimg_out_file)

        mask_out = Image.fromarray(mask)
        mask_out.save(mask_out_file)

    # Save the image
    img_out_file = os.path.join(img_dir, "image.png")

    img_out = Image.fromarray(img)
    img_out.save(img_out_file)

    return img_dir, mask_dir

def save_cropped_img_mask(big_img_file, big_mask_file, output_dir, create_number_per_image, fname):
    # Create folders for output
    output_QA_dir = os.path.join(output_dir,'QA_roi')
    output_img_dir = os.path.join(output_dir,'train_roi')
    output_mask_dir = os.path.join(output_dir,'labels_roi')

    if not os.path.exists(output_QA_dir):
        os.makedirs(output_QA_dir)
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
    if not os.path.exists(output_mask_dir):
        os.makedirs(output_mask_dir)

    roi_count = 0

    # Get a list of ROI
    big_mask = Image.open(big_mask_file)
    big_mask_binary = np.array(big_mask.convert('L'))

    big_img = Image.open(big_img_file)
    big_img_arr = np.array(big_img)

    labels = label(big_mask_binary)

if __name__ == "__main__":
    # Define the directories for the input and output
    # png_file = "/home/sybbure/CircleNet/MoNuSeg Training Data/Tissue Images/TCGA-18-5592-01Z-00-DX1.png"
    # xml_file = "/home/sybbure/CircleNet/MoNuSeg Training Data/Annotations/TCGA-18-5592-01Z-00-DX1.xml"

    input_dir = '/home/sybbure/CircleNet/MoNuSeg Training Data/Tissue Images'
    label_dir = '/home/sybbure/CircleNet/MoNuSeg Training Data/Annotations/'
    output_dir = "/home/sybbure/CircleNet/MoNuSeg-Test/coco-mask"

    create_number_per_image = 10
    img_size = [1000, 1000]

    # Obtain list of files
    png_files = []
    for (root, dirs, files) in os.walk(input_dir):
        png_files = files
        break

    # Iterate over images
    for file in png_files:
        png_file = os.path.join(input_dir, file)
        xml_file = (os.path.join(label_dir, file)).replace('.png', '.xml')

        # Get the big img and mask
        big_img, big_mask = png_to_big_mask(png_file, xml_file, output_dir)

        # Data augmentation - crop the image and save it
        # save_cropped_img_mask(big_img, big_mask, output_dir, create_number_per_image, file)


