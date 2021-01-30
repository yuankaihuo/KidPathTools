import xmltodict
import os
import numpy as np
from PIL import Image
from skimage.measure import label
import cv2
import random

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
    output_img_dir = os.path.join(output_dir,"whole", file.split(".")[0])

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
        if ci % 100 == 0:
            print(".", end="")
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

        # Draw masks (note - color is white and thickness = -1)
        cv2.drawContours(mask, [cnt.astype(int)], -1, (255, 255, 255), -1)

        mask_name = "mask_%03d.png" % ci
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

    print()

    return img_out_file, mask_dir

def save_cropped_img_mask(big_img_file, big_mask_dir, output_dir, create_number_per_image, fname, crop_size=(512, 512)):
    # Create folders for output
    # output_QA_dir = os.path.join(output_dir,'QA_roi')
    # output_img_dir = os.path.join(output_dir,'train_roi')
    # output_mask_dir = os.path.join(output_dir,'labels_roi')
    output_roi_dir = os.path.join(output_dir, "roi")

    # if not os.path.exists(output_QA_dir):
    #     os.makedirs(output_QA_dir)
    # if not os.path.exists(output_img_dir):
    #     os.makedirs(output_img_dir)
    # if not os.path.exists(output_mask_dir):
    #     os.makedirs(output_mask_dir)

    if not os.path.exists(output_roi_dir):
        os.makedirs(output_roi_dir)

    img = np.array(Image.open(os.path.join(big_img_file)))

    height = crop_size[0]
    width = crop_size[1]

    for i in range(create_number_per_image):
        print('.',end='')

        assert img.shape[0] >= height
        assert img.shape[1] >= width

        # Random crop
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        crop_img = img[y:y + height, x:x + width]

        crop_img_name = '%s-contour%03d-x-%d-y-%d' % (fname.split(".")[0], i, x, y)

        dir = os.path.join(output_roi_dir, crop_img_name)
        if not os.path.exists(dir):
            os.makedirs(dir)

        crop_img_dir = os.path.join(dir, 'image')
        if not os.path.exists(crop_img_dir):
            os.makedirs(crop_img_dir)

        crop_mask_dir = os.path.join(dir, 'masks')
        if not os.path.exists(crop_mask_dir):
            os.makedirs(crop_mask_dir)

        crop_img_path = os.path.join(crop_img_dir, crop_img_name + '.png')
        img_out = Image.fromarray(crop_img)
        img_out.save(crop_img_path)

        for (root, dirs, files) in os.walk(big_mask_dir):
            mask_files = files
            break

        num_mask = 0
        for mask_file in mask_files:
            mask_file_path = os.path.join(big_mask_dir, mask_file)
            mask = Image.open(mask_file_path)
            mask = np.array(mask)

            # Crop the mask
            crop_mask = mask[y:y + height, x:x + width]
            assert crop_img.shape[0] == crop_mask.shape[0]
            assert crop_img.shape[1] == crop_mask.shape[1]

            # If mask is not empty, then save it
            if max(crop_mask.ravel()) != 0:
                mask_out = os.path.join(crop_mask_dir, "mask%04d.png" % (num_mask))

                crop_mask = Image.fromarray(mask)
                crop_mask.save(mask_out)

                num_mask += 1
    print()
    return

if __name__ == "__main__":
    # input_dir = '/home/sybbure/CircleNet/MoNuSegTestData'
    # # Obtain list of files
    # files = []
    # for (root, dirs, files) in os.walk(input_dir):
    #     files = files
    #     break
    # for file in files:
    #     format = file.split('.')[-1]
    #     subname = file.split('.')[0]
    #     print(format)
    #     if format == 'tif':
    #         cur_image_path = os.path.join(input_dir, file)
    #         new_image_dir = os.path.join(input_dir, "images", file)
    #         os.system("cp '%s' '%s'" % (cur_image_path, new_image_dir))
    #
    #     elif format == 'xml':
    #         cur_image_path = os.path.join(input_dir, file)
    #         new_image_dir = os.path.join(input_dir, "xml", file)
    #         os.system("cp '%s' '%s'" % (cur_image_path, new_image_dir))


    # Define the directories for the input and output
    # png_file = "/home/sybbure/CircleNet/MoNuSeg Training Data/Tissue Images/TCGA-18-5592-01Z-00-DX1.png"
    # xml_file = "/home/sybbure/CircleNet/MoNuSeg Training Data/Annotations/TCGA-18-5592-01Z-00-DX1.xml"

    input_dir = '/home/sybbure/CircleNet/MoNuSegTestData/images'
    label_dir = '/home/sybbure/CircleNet/MoNuSegTestData/xml'
    output_dir = '/home/sybbure/CircleNet/MoNuSeg-Test/png-instance-segmentation'

    create_number_per_image = 10
    img_size = [1000, 1000]

    # Obtain list of files
    png_files = []
    for (root, dirs, files) in os.walk(input_dir):
        png_files = files
        break

    # png_files = png_files[0:2]
    # print(png_files)

    # Iterate over images
    for file in png_files:
        print(file)
        png_file = os.path.join(input_dir, file)
        # xml_file = (os.path.join(label_dir, file)).replace('.png', '.xml')
        xml_file = (os.path.join(label_dir, file)).replace('.tif', '.xml')
        # Get the big img and mask
        big_img, big_mask = png_to_big_mask(png_file, xml_file, output_dir)

        # Data augmentation - crop the image and save it
        save_cropped_img_mask(big_img, big_mask, output_dir, create_number_per_image, file)