import openslide
import xmltodict
import numpy as np
from PIL import Image
import os
import cv2
import random
from skimage.measure import label
from libtiff import TIFF

def cacluate_equal_size(img_size, pixel_size, mpp_x, level_downsamples):
    micron_patch = pixel_size * img_size[0]
    for li in range(len(level_downsamples)):
        level = level_downsamples[li]
        new_res = micron_patch / mpp_x / level
        if new_res < img_size[0]-1:
            lv = li - 1
            break

    final_res = int(np.round(micron_patch / mpp_x / level_downsamples[lv]))
    res_size = [final_res, final_res]
    return res_size, lv


def scn_to_png_atubular(simg,xml_file,output_dir, create_number_per_image, img_size, fname, pixel_size, auto_res_size=False):
    # read region
    with open(xml_file) as fd:
        doc = xmltodict.parse(fd.read())
    layers = doc['Annotations']['Annotation']

    random.seed(0)

    try:
        start_x = np.int(simg.properties['openslide.bounds-x'])
        start_y = np.int(simg.properties['openslide.bounds-y'])
        width_x = np.int(simg.properties['openslide.bounds-width'])
        height_y = np.int(simg.properties['openslide.bounds-height'])
    except:
        start_x = 0
        start_y = 0
        width_x = np.int(simg.properties['aperio.OriginalWidth'])
        height_y = np.int(simg.properties['aperio.OriginalHeight'])
    end_x = start_x + width_x
    end_y = start_y + height_y



    mpp_x = float(simg.properties['openslide.mpp-x']) #micron per pixel x
    mpp_y = float(simg.properties['openslide.mpp-y']) #micron per pixel y
    assert mpp_x == mpp_y
    level_downsamples = simg.level_downsamples

    if auto_res_size: # caculate auto res size
        res_size, lv = cacluate_equal_size(img_size, pixel_size, mpp_x, level_downsamples)
    else:
        res_size0, lv = cacluate_equal_size(img_size, pixel_size, mpp_x, level_downsamples)
        res_size = img_size


    output_QA_dir = os.path.join(output_dir,'QA')
    output_img_dir = os.path.join(output_dir,'train')
    output_mask_dir = os.path.join(output_dir,'labels')

    if not os.path.exists(output_QA_dir):
        os.makedirs(output_QA_dir)
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
    if not os.path.exists(output_mask_dir):
        os.makedirs(output_mask_dir)

    if isinstance(layers, (dict)):
        layers = [layers]

    for i in range(len(layers)):
        regions = layers[i]['Regions']

        if isinstance(layers[i]['Attributes'], dict):
            clss_name = layers[i]['Attributes']['Attribute']['@Name']
        else:
            clss_name = 'unknown'

        if (len(regions) < 2):
            notFound = layers[0]
        else:
            regions = regions['Region']

            if isinstance(regions, (dict)):
                regions = [regions]

            for j in range(len(regions)):
                contours = regions[j]

                for ri in range(0, create_number_per_image):
                    img, cimg, mask, x0_rand, y0_rand = get_contour_atubular(simg, contours, start_x, start_y, width_x, height_y,
                                                                    img_size, res_size, lv)
                    png_name = '%s-x-contour%03d-x-%d-x-%d.png' % (fname, 0, x0_rand, y0_rand)

                    img_all = np.concatenate((img, cimg, mask), axis=1)
                    img_all_out = Image.fromarray(img_all)
                    img_all_out_file = os.path.join(output_QA_dir, png_name)
                    img_all_out.save(img_all_out_file)

                    img_out = Image.fromarray(img)
                    img_out_file = os.path.join(output_img_dir, png_name)
                    img_out.save(img_out_file)

                    mask_out = Image.fromarray(mask)
                    mask_out_file = os.path.join(output_mask_dir, png_name)
                    mask_out.save(mask_out_file)


def scn_to_png(simg,xml_file,output_dir, create_number_per_image, img_size, fname):

    # read region
    with open(xml_file) as fd:
        doc = xmltodict.parse(fd.read())
    layers = doc['Annotations']['Annotation']
    try :
        contours = layers['Regions']['Region']
    except:
        if len(layers) == 2:
            BBlayer = layers[0]
            regions = BBlayer['Regions']['Region']
            Masklayer = layers[1]
        else:
            Masklayer = layers[0]
        contours = Masklayer['Regions']['Region']

    start_x, start_y = get_nonblack_starting_point(simg)

    output_QA_dir = os.path.join(output_dir,'QA')
    output_img_dir = os.path.join(output_dir,'train')
    output_mask_dir = os.path.join(output_dir,'labels')

    if not os.path.exists(output_QA_dir):
        os.makedirs(output_QA_dir)
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
    if not os.path.exists(output_mask_dir):
        os.makedirs(output_mask_dir)

    for ri in range(0,create_number_per_image):

        try:
            contours['Vertices']
            img, cimg, mask, x0_rand, y0_rand = get_contour(simg, contours, start_x, start_y, create_number_per_image, img_size)
            png_name = '%s-x-contour%03d-x-%d-x-%d.png' % (fname, 0, x0_rand, y0_rand)

            img_all = np.concatenate((img, cimg, mask), axis=1)
            img_all_out = Image.fromarray(img_all)
            img_all_out_file = os.path.join(output_QA_dir, png_name)
            img_all_out.save(img_all_out_file)

            img_out = Image.fromarray(img)
            img_out_file = os.path.join(output_img_dir, png_name)
            img_out.save(img_out_file)

            mask_out = Image.fromarray(mask)
            mask_out_file = os.path.join(output_mask_dir, png_name)
            mask_out.save(mask_out_file)

        except:
            for i in range(len(contours)):
                contour = contours[i]

                img, cimg, mask, x0_rand, y0_rand = get_contour(simg,  contour, start_x, start_y, create_number_per_image, img_size)

                png_name = '%s-x-contour%03d-x-%d-x-%d.png'%(fname,i,x0_rand,y0_rand)

                img_all = np.concatenate((img, cimg, mask), axis=1)
                img_all_out = Image.fromarray(img_all)
                img_all_out_file = os.path.join(output_QA_dir, png_name)
                img_all_out.save(img_all_out_file)

                img_out = Image.fromarray(img)
                img_out_file = os.path.join(output_img_dir, png_name)
                img_out.save(img_out_file)

                mask_out = Image.fromarray(mask)
                mask_out_file = os.path.join(output_mask_dir, png_name)
                mask_out.save(mask_out_file)

def scn_to_png_whole_slide_good_bad(simg,xml_file,output_dir, create_number_per_image, img_size, pixel_size, pixel_size_threshold, fname, read_patch=False, auto_res_size=False):
    # read region
    with open(xml_file) as fd:
        doc = xmltodict.parse(fd.read())
    layers = doc['Annotations']['Annotation']

    random.seed(0)

    try:
        start_x = np.int(simg.properties['openslide.bounds-x'])
        start_y = np.int(simg.properties['openslide.bounds-y'])
        width_x = np.int(simg.properties['openslide.bounds-width'])
        height_y = np.int(simg.properties['openslide.bounds-height'])
    except:
        start_x = 0
        start_y = 0
        width_x = np.int(simg.properties['aperio.OriginalWidth'])
        height_y = np.int(simg.properties['aperio.OriginalHeight'])
    end_x = start_x + width_x
    end_y = start_y + height_y


    mpp_x = float(simg.properties['openslide.mpp-x']) #micron per pixel x
    mpp_y = float(simg.properties['openslide.mpp-y']) #micron per pixel y
    assert mpp_x == mpp_y
    level_downsamples = simg.level_downsamples

    if auto_res_size: # caculate auto res size
        res_size, lv = cacluate_equal_size(img_size, pixel_size, mpp_x, level_downsamples)
    else:
        res_size0, lv = cacluate_equal_size(img_size, pixel_size, mpp_x, level_downsamples)
        res_size = img_size


    output_QA_dir = os.path.join(output_dir,'QA_whole')
    output_img_dir = os.path.join(output_dir,'train_whole')
    output_mask_dir = os.path.join(output_dir,'labels_whole')
    output_bad_mask_dir = os.path.join(output_dir, 'bad_labels_whole')

    if not os.path.exists(output_QA_dir):
        os.makedirs(output_QA_dir)
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
    if not os.path.exists(output_mask_dir):
        os.makedirs(output_mask_dir)
    if not os.path.exists(output_bad_mask_dir):
        os.makedirs(output_bad_mask_dir)

    png_name = '%s-x-whole_slice.png' % (fname)
    img_all_out_file = os.path.join(output_QA_dir, png_name)
    img_out_file = os.path.join(output_img_dir, png_name)
    mask_out_file = os.path.join(output_mask_dir, png_name)
    bad_mask_out_file = os.path.join(output_bad_mask_dir, png_name)

    if os.path.exists(img_all_out_file) and os.path.exists(img_out_file) and os.path.exists(mask_out_file):
        return img_out_file, mask_out_file

    if isinstance(layers, (dict)):
        layers = [layers]

    img, cimg, mask, bad_mask = get_whole_slide_good_bad(simg,  layers, start_x, start_y, width_x, height_y, pixel_size, pixel_size_threshold, lv, read_patch)

    # img_all = np.concatenate((img, cimg, mask), axis=1)
    # img_all_out = Image.fromarray(img_all)
    # img_all_out.save(img_all_out_file)

    img_out = Image.fromarray(img)
    img_out.save(img_out_file)

    mask_out = Image.fromarray(mask.astype(np.uint8))
    mask_out.save(mask_out_file)

    # bad_mask_out = Image.fromarray(bad_mask.astype(np.uint8))
    # bad_mask_out.save(bad_mask_out_file)

    return img_out_file, mask_out_file, img, mask, bad_mask


def scn_to_png_whole_slide_atubular(simg,xml_file,output_dir, create_number_per_image, img_size, pixel_size, pixel_size_threshold, fname, read_patch=False, auto_res_size=False):
    # read region
    with open(xml_file) as fd:
        doc = xmltodict.parse(fd.read())
    layers = doc['Annotations']['Annotation']

    random.seed(0)

    try:
        start_x = np.int(simg.properties['openslide.bounds-x'])
        start_y = np.int(simg.properties['openslide.bounds-y'])
        width_x = np.int(simg.properties['openslide.bounds-width'])
        height_y = np.int(simg.properties['openslide.bounds-height'])
    except:
        start_x = 0
        start_y = 0
        width_x = np.int(simg.properties['aperio.OriginalWidth'])
        height_y = np.int(simg.properties['aperio.OriginalHeight'])
    end_x = start_x + width_x
    end_y = start_y + height_y


    mpp_x = float(simg.properties['openslide.mpp-x']) #micron per pixel x
    mpp_y = float(simg.properties['openslide.mpp-y']) #micron per pixel y
    assert mpp_x == mpp_y
    level_downsamples = simg.level_downsamples

    if auto_res_size: # caculate auto res size
        res_size, lv = cacluate_equal_size(img_size, pixel_size, mpp_x, level_downsamples)
    else:
        res_size0, lv = cacluate_equal_size(img_size, pixel_size, mpp_x, level_downsamples)
        res_size = img_size


    output_QA_dir = os.path.join(output_dir,'QA_whole')
    output_img_dir = os.path.join(output_dir,'train_whole')
    output_mask_dir = os.path.join(output_dir,'labels_whole')

    if not os.path.exists(output_QA_dir):
        os.makedirs(output_QA_dir)
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
    if not os.path.exists(output_mask_dir):
        os.makedirs(output_mask_dir)

    png_name = '%s-x-whole_slice.png' % (fname)
    img_all_out_file = os.path.join(output_QA_dir, png_name)
    img_out_file = os.path.join(output_img_dir, png_name)
    mask_out_file = os.path.join(output_mask_dir, png_name)

    if os.path.exists(img_all_out_file) and os.path.exists(img_out_file) and os.path.exists(mask_out_file):
        return img_out_file, mask_out_file

    if isinstance(layers, (dict)):
        layers = [layers]

    img, cimg, mask = get_whole_slide_atubular(simg,  layers, start_x, start_y, width_x, height_y, pixel_size, pixel_size_threshold, lv, read_patch)

    # img_all = np.concatenate((img, cimg, mask), axis=1)
    # img_all_out = Image.fromarray(img_all)
    # img_all_out.save(img_all_out_file)

    img_out = Image.fromarray(img)
    img_out.save(img_out_file)

    mask_out = Image.fromarray(mask)
    mask_out.save(mask_out_file)

    return img_out_file, mask_out_file, img, mask

def scn_to_png_whole_slide(simg,xml_file,output_dir, create_number_per_image, img_size, fname):

    # read region
    with open(xml_file) as fd:
        doc = xmltodict.parse(fd.read())
    layers = doc['Annotations']['Annotation']
    try :
        contours = layers['Regions']['Region']
    except:
        if len(layers) == 2:
            BBlayer = layers[0]
            regions = BBlayer['Regions']['Region']
            Masklayer = layers[1]
        else:
            Masklayer = layers[0]
        contours = Masklayer['Regions']['Region']

    #check if finish all files
    output_QA_dir = os.path.join(output_dir,'QA_whole')
    output_img_dir = os.path.join(output_dir,'train_whole')
    output_mask_dir = os.path.join(output_dir,'labels_whole')

    png_name = '%s-x-whole_slice.png' % (fname)
    img_all_out_file = os.path.join(output_QA_dir, png_name)
    img_out_file = os.path.join(output_img_dir, png_name)
    mask_out_file = os.path.join(output_mask_dir, png_name)

    if os.path.exists(img_all_out_file) and os.path.exists(img_out_file) and os.path.exists(mask_out_file):
        return img_out_file, mask_out_file

    start_x, start_y = get_nonblack_starting_point(simg)
    end_x, end_y = get_nonblack_ending_point(simg)

    # simg.read_region((end_x, end_y), 3, (1000, 1000)).show()
    # simg.read_region((start_x, start_y), 3, (end_x-start_x, end_y-start_y)).show()



    if not os.path.exists(output_QA_dir):
        os.makedirs(output_QA_dir)
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
    if not os.path.exists(output_mask_dir):
        os.makedirs(output_mask_dir)



    try:
        contours['Vertices']
        img, cimg, mask = get_whole_slide(simg, [contours], start_x, start_y, end_x, end_y, create_number_per_image, img_size)
    except:
        img, cimg, mask = get_whole_slide(simg,  contours, start_x, start_y, end_x, end_y, create_number_per_image, img_size)

    img_all = np.concatenate((img, cimg, mask), axis=1)
    img_all_out = Image.fromarray(img_all)
    img_all_out.save(img_all_out_file)

    img_out = Image.fromarray(img)
    img_out.save(img_out_file)

    mask_out = Image.fromarray(mask)
    mask_out.save(mask_out_file)

    return img_out_file, mask_out_file


def get_none_zero(black_arr):

    nonzeros = black_arr.nonzero()
    starting_y = nonzeros[0].min()
    ending_y = nonzeros[0].max()
    starting_x = nonzeros[1].min()
    ending_x = nonzeros[1].max()

    return starting_x, starting_y, ending_x, ending_y

def scan_nonblack(simg,px_start,py_start,px_end,py_end):
    offset_x = 0
    offset_y = 0
    line_x = py_end-py_start
    line_y = px_end-px_start

    val = simg.read_region((px_start+offset_x, py_start), 0, (1, 1))
    arr = np.array(val)[:, :, 0].sum()
    while arr == 0:
        val = simg.read_region((px_start+offset_x, py_start), 0, (1, line_x))
        arr = np.array(val)[:, :, 0].sum()
        offset_x = offset_x + 1

    val = simg.read_region((px_start, py_start+offset_y), 0, (1, 1))
    arr = np.array(val)[:, :, 0].sum()
    while arr == 0:
        val = simg.read_region((px_start, py_start+offset_y), 0, (line_y, 1))
        arr = np.array(val)[:, :, 0].sum()
        offset_y = offset_y + 1

    x = px_start+offset_x-1
    y = py_start+offset_y-1
    return x,y


def scan_nonblack_end(simg,px_start,py_start,px_end,py_end):
    offset_x = 0
    offset_y = 0
    line_x = py_end-py_start
    line_y = px_end-px_start

    val = simg.read_region((px_end+offset_x, py_end), 0, (1, 1))
    arr = np.array(val)[:, :, 0].sum()
    while not arr == 0:
        val = simg.read_region((px_end+offset_x, py_end), 0, (1, line_x))
        arr = np.array(val)[:, :, 0].sum()
        offset_x = offset_x + 1

    val = simg.read_region((px_end, py_end+offset_y), 0, (1, 1))
    arr = np.array(val)[:, :, 0].sum()
    while not arr == 0:
        val = simg.read_region((px_end, py_end+offset_y), 0, (line_y, 1))
        arr = np.array(val)[:, :, 0].sum()
        offset_y = offset_y + 1

    x = px_end+(offset_x-1)
    y = py_end+(offset_y-1)
    return x,y


def get_nonblack_starting_point(simg):
    px = 0
    py = 0
    black_img = simg.read_region((px, py), 3, (3000, 3000))
    starting_x, starting_y, ending_x, ending_y = get_none_zero(np.array(black_img)[:, :, 0])

    multiples = int(np.floor(simg.level_dimensions[0][0]/float(simg.level_dimensions[3][0])))

    #staring point
    px2 = (starting_x - 1) * multiples
    py2 = (starting_y - 1) * multiples
    #ending point
    px3 = (ending_x + 1) * multiples
    py3 = (ending_y + 1) * multiples

    # black_img_big = simg.read_region((px2, py2), 0, (1000, 1000))
    # offset_x, offset_y, offset_xx, offset_yy = get_none_zero(np.array(black_img_big)[:, :, 0])
    #
    # x = px2+offset_x
    # y = py2+offset_y

    xx, yy = scan_nonblack(simg, px2, py2, px3, py3)

    return xx,yy

def get_nonblack_ending_point(simg):
    px = 0
    py = 0
    black_img = simg.read_region((px, py), 3, (3000, 3000))
    starting_x, starting_y, ending_x, ending_y = get_none_zero(np.array(black_img)[:, :, 0])

    multiples = int(np.floor(simg.level_dimensions[0][0]/float(simg.level_dimensions[3][0])))

    #staring point
    px2 = (starting_x - 1) * multiples
    py2 = (starting_y - 1) * multiples
    #ending point
    px3 = (ending_x - 1) * (multiples-1)
    py3 = (ending_y - 1) * (multiples-1)

    # black_img_big = simg.read_region((px2, py2), 0, (1000, 1000))
    # offset_x, offset_y, offset_xx, offset_yy = get_none_zero(np.array(black_img_big)[:, :, 0])
    #
    # x = px2+offset_x
    # y = py2+offset_y

    xx, yy = scan_nonblack_end(simg, px2, py2, px3, py3)

    return xx,yy


def get_ROI(simg, region):
    vertices = region['Vertices']['Vertex']
    x0 = float(vertices[0]['@X'])
    y0 = float(vertices[0]['@Y'])
    z0 = float(vertices[0]['@Z'])
    x1 = float(vertices[1]['@X'])
    y1 = float(vertices[1]['@Y'])
    z1 = float(vertices[1]['@Z'])
    x2 = float(vertices[2]['@X'])
    y2 = float(vertices[2]['@Y'])
    z2 = float(vertices[2]['@Z'])
    x3 = float(vertices[3]['@X'])
    y3 = float(vertices[3]['@Y'])
    z3 = float(vertices[3]['@Z'])

    # get manual ROI coordinate
    ys = int(round(x0))
    xs = int(round(y0))
    yss = int(round(x1))
    xss = int(round(y2))
    widths = int(round(x1 - x0))
    heights = int(round(y2 - y1))

    start_x, start_y = get_nonblack_starting_point(simg)
    #isimg.read_region((44600, 82700), 0, (widths,heights).show()
    max_widths = int(simg.properties['openslide.bounds-height'])
    img = simg.read_region((start_x+xs, max_widths-yss+start_y), 0, (heights,widths))
    img = np.array(img.convert('RGB'))
    return img


def get_contour_atubular(simg, contour, start_x, start_y, width_x, height_y, img_size, res_size, lv):
    if 'leica.device-model' in simg.properties:
        max_height = int(width_x)
        max_widths = int(height_y)
        vertices = contour['Vertices']['Vertex']
        x_min = max_height
        x_max = 0
        y_min = max_widths
        y_max = 0

        for vi in range(len(vertices)):
            xraw = float(vertices[vi]['@Y'])
            yraw = float(vertices[vi]['@X'])
            if xraw < x_min:
                x_min = xraw
            if xraw > x_max:
                x_max = xraw
            if yraw < y_min:
                y_min = yraw
            if yraw > y_max:
                y_max = yraw

            x_min = int(round(x_min))
            x_max = int(round(x_max))
            y_min = int(round(y_min))
            y_max = int(round(y_max))

            # add cropping
            xx_min = max(x_min-50, 0)
            xx_max = min(x_max+50, max_height)
            yy_min = max(y_min-50, 0)
            yy_max = min(y_max+50, max_widths)

            heights = xx_max-xx_min
            widths = yy_max-yy_min

            xs = xx_min
            yss = yy_max


        cnt = np.zeros((len(vertices),1,2))
        for vi in range(len(vertices)):
            xx = float(vertices[vi]['@Y'])-xs
            yy = yss - float(vertices[vi]['@X'])
            cnt[vi,0,0] = int(xx)
            cnt[vi,0,1] = int(yy)



        #isimg.read_region((44600, 82700), 0, (widths,heights).show()
        down_rate = simg.level_downsamples[lv]
        resolution = simg.level_dimensions[lv]

        heights_lv2 = int(np.ceil(heights/down_rate))
        widths_lv2 = int(np.ceil(widths/down_rate))


        x0 = start_x+xs
        y0 = max_widths-yss+start_y

        rand_minx = 0
        rand_miny = 0
        rand_maxx = res_size[0] - heights_lv2 - 1
        rand_mayy = res_size[1] - widths_lv2 - 1

        x0_rand = x0 - int(random.randint(0, rand_maxx) * np.round(down_rate))
        y0_rand = y0 - int(random.randint(0, rand_mayy) * np.round(down_rate))

        x0_rand = min(x0_rand, simg.level_dimensions[0][0] - res_size[0] - 1)
        x0_rand = max(x0_rand, 0)
        y0_rand = min(y0_rand, simg.level_dimensions[0][1] - res_size[1] - 1)
        y0_rand = max(y0_rand, 0)

        img = simg.read_region((x0_rand, y0_rand), lv, (res_size[0], res_size[1]))
        img = np.array(img.convert('RGB'))

        cimg = img.copy()
        xs2 = x0_rand - start_x
        yss2 = max_widths - start_y - y0_rand
        vertices = contour['Vertices']['Vertex']
        cnt = np.zeros((len(vertices), 1, 2))
        for vi in range(len(vertices)):
            xx = float(vertices[vi]['@Y']) - (xs2)
            yy = yss2 - float(vertices[vi]['@X'])
            cnt[vi, 0, 0] = int(xx / down_rate)
            cnt[vi, 0, 1] = int(yy / down_rate)

    elif 'aperio.Filename' in simg.properties:
        max_height = int(height_y)
        max_widths = int(width_x)
        vertices = contour['Vertices']['Vertex']
        x_min = max_widths
        x_max = 0
        y_min = max_height
        y_max = 0

        for vi in range(len(vertices)):
            xraw = float(vertices[vi]['@X'])
            yraw = float(vertices[vi]['@Y'])
            if xraw < x_min:
                x_min = xraw
            if xraw > x_max:
                x_max = xraw
            if yraw < y_min:
                y_min = yraw
            if yraw > y_max:
                y_max = yraw

            x_min = int(round(x_min))
            x_max = int(round(x_max))
            y_min = int(round(y_min))
            y_max = int(round(y_max))

            # add cropping
            xx_min = max(x_min - 50, 0)
            xx_max = min(x_max + 50, max_widths)
            yy_min = max(y_min - 50, 0)
            yy_max = min(y_max + 50, max_height)

            widths = xx_max - xx_min
            heights = yy_max - yy_min

            xs = xx_min
            yss = yy_min

        cnt = np.zeros((len(vertices), 1, 2))
        for vi in range(len(vertices)):
            xx = float(vertices[vi]['@X']) - xs
            yy = float(vertices[vi]['@Y']) - yss
            cnt[vi, 0, 0] = int(xx)
            cnt[vi, 0, 1] = int(yy)

        # isimg.read_region((44600, 82700), 0, (widths,heights).show()
        down_rate = simg.level_downsamples[lv]
        resolution = simg.level_dimensions[lv]

        heights_lv = int(np.ceil(heights / down_rate))
        widths_lv = int(np.ceil(widths / down_rate))

        x0 = xs  + start_x
        y0 = yss + start_y

        rand_minx = 0
        rand_miny = 0
        rand_maxx = res_size[0] - widths_lv - 1
        rand_mayy = res_size[1] - heights_lv - 1

        x0_rand = x0 - int(random.randint(0, rand_maxx) * np.round(down_rate))
        y0_rand = y0 - int(random.randint(0, rand_mayy) * np.round(down_rate))

        x0_rand = min(x0_rand, simg.level_dimensions[0][0] - res_size[0] - 1)
        x0_rand = max(x0_rand, 0)
        y0_rand = min(y0_rand, simg.level_dimensions[0][1] - res_size[1] - 1)
        y0_rand = max(y0_rand, 0)


        # simg.read_region((x0_rand, y0_rand), 2, (res_size[0], res_size[1])).show()

        img = simg.read_region((x0_rand, y0_rand), lv, (res_size[0],res_size[1]))
        img = np.array(img.convert('RGB'))
        # Image.fromarray(img).show()

        cimg = img.copy()
        xs2 = x0_rand-start_x
        yss2 = y0_rand- start_y
        vertices = contour['Vertices']['Vertex']
        cnt = np.zeros((len(vertices),1,2))
        for vi in range(len(vertices)):
            xx = float(vertices[vi]['@X'])-(xs2)
            yy = float(vertices[vi]['@Y'])-(yss2)
            cnt[vi,0,0] = int(xx/down_rate)
            cnt[vi,0,1] = int(yy/down_rate)


    cv2.drawContours(cimg, [cnt.astype(int)], -1, (0, 255, 0), 1)

    #draw mask
    mask = np.zeros(cimg.shape, dtype=np.uint8)
    cv2.drawContours(mask, [cnt.astype(int)], -1, (255, 255, 255), -1)

    # Image.fromarray(cimg).show()


    return img, cimg, mask, x0_rand, y0_rand




def get_contour(simg, contour, start_x, start_y, create_number_per_image, img_size):
    max_height = int(simg.properties['openslide.bounds-width'])
    max_widths = int(simg.properties['openslide.bounds-height'])
    vertices = contour['Vertices']['Vertex']
    x_min = max_height
    x_max = 0
    y_min = max_widths
    y_max = 0



    for vi in range(len(vertices)):
        xraw = float(vertices[vi]['@Y'])
        yraw = float(vertices[vi]['@X'])
        if xraw < x_min:
            x_min = xraw
        if xraw > x_max:
            x_max = xraw
        if yraw < y_min:
            y_min = yraw
        if yraw > y_max:
            y_max = yraw

        x_min = int(round(x_min))
        x_max = int(round(x_max))
        y_min = int(round(y_min))
        y_max = int(round(y_max))

        # add cropping
        xx_min = max(x_min-50, 0)
        xx_max = min(x_max+50, max_height)
        yy_min = max(y_min-50, 0)
        yy_max = min(y_max+50, max_widths)

        heights = xx_max-xx_min
        widths = yy_max-yy_min

        xs = xx_min
        yss = yy_max


    cnt = np.zeros((len(vertices),1,2))
    for vi in range(len(vertices)):
        xx = float(vertices[vi]['@Y'])-xs
        yy = yss - float(vertices[vi]['@X'])
        cnt[vi,0,0] = int(xx)
        cnt[vi,0,1] = int(yy)



    #isimg.read_region((44600, 82700), 0, (widths,heights).show()
    lv = 2
    down_rate = simg.level_downsamples[lv]
    resolution = simg.level_dimensions[lv]

    heights_lv2 = int(np.ceil(heights/down_rate))
    widths_lv2 = int(np.ceil(widths/down_rate))

    x0 = start_x+xs
    y0 = max_widths-yss+start_y

    rand_minx = 0
    rand_miny = 0
    rand_maxx = img_size[0] - heights_lv2 - 1
    rand_mayy = img_size[1] - widths_lv2 - 1

    x0_rand = x0 - int(random.randint(0,rand_maxx)*np.round(down_rate))
    y0_rand = y0 - int(random.randint(0,rand_mayy)*np.round(down_rate))

    x0_rand = min(x0_rand, simg.level_dimensions[0][0] - img_size[0] - 1)
    x0_rand = max(x0_rand, 0)
    y0_rand = min(y0_rand, simg.level_dimensions[0][1] - img_size[1] - 1)
    y0_rand = max(y0_rand, 0)

    # simg.read_region((x0_rand, y0_rand), 2, (img_size[0], img_size[1])).show()

    img = simg.read_region((x0_rand, y0_rand), 2, (img_size[0],img_size[1]))
    img = np.array(img.convert('RGB'))

    cimg = img.copy()
    xs2 = x0_rand-start_x
    yss2 = max_widths  + start_y -y0_rand
    vertices = contour['Vertices']['Vertex']
    cnt = np.zeros((len(vertices),1,2))
    for vi in range(len(vertices)):
        xx = float(vertices[vi]['@Y'])-(xs2)
        yy = yss2 - float(vertices[vi]['@X'])
        cnt[vi,0,0] = int(xx/down_rate)
        cnt[vi,0,1] = int(yy/down_rate)

    cv2.drawContours(cimg, [cnt.astype(int)], -1, (0, 255, 0), 1)

    #draw mask
    mask = np.zeros(cimg.shape, dtype=np.uint8)
    cv2.drawContours(mask, [cnt.astype(int)], -1, (255, 255, 255), -1)

    # Image.fromarray(cimg).show()


    return img, cimg, mask, x0_rand, y0_rand


def get_whole_slide_atubular(simg, layers, start_x, start_y, width_x, height_y, pixel_size, pixel_size_threshold, lv, read_patch = False):
    if 'leica.device-model' in simg.properties:
        down_rate = simg.level_downsamples[lv]
        end_x = start_x + width_x
        end_y = start_y + height_y


        max_height = width_x
        max_widths = height_y

        whole_height_lv2 = int(np.ceil((end_x - start_x) / down_rate))
        whole_width_lv2 = int(np.ceil((end_y - start_y) / down_rate))
        img = simg.read_region((start_x, start_y), lv, (whole_height_lv2, whole_width_lv2))
        img = np.array(img.convert('RGB'))
        cimg = img.copy()
        mask = np.zeros(cimg.shape, dtype=np.uint8)

        for i in range(len(layers)):
            regions = layers[i]['Regions']

            if isinstance(layers[i]['Attributes'], dict):
                clss_name = layers[i]['Attributes']['Attribute']['@Name']
            else:
                clss_name = 'unknown'

            if (len(regions) < 2):
                notFound = layers[0]
            else:
                regions = regions['Region']

                if isinstance(regions, (dict)):
                    regions = [regions]

                for j in range(len(regions)):
                    contour = regions[j]

                    vertices = contour['Vertices']['Vertex']

                    if len(vertices) <= 4:
                        continue

                    cnt = np.zeros((len(vertices),1,2))
                    for vi in range(len(vertices)):
                        xx = float(vertices[vi]['@Y'])
                        yy = max_widths - float(vertices[vi]['@X'])
                        cnt[vi,0,0] = int(xx/down_rate)
                        cnt[vi,0,1] = int(yy/down_rate)

                    # cv2.contourArea(cnt)

                    #draw contour
                    cv2.drawContours(cimg, [cnt.astype(int)], -1, (0, 255, 0), 1)

                    #draw masks
                    cv2.drawContours(mask, [cnt.astype(int)], -1, (255, 255, 255), -1)

    elif 'aperio.Filename' in simg.properties:
        down_rate = simg.level_downsamples[lv]
        end_x = start_x + width_x
        end_y = start_y + height_y

        max_height = height_y
        max_widths = width_x

        whole_height_lv2 = int(np.ceil(height_y / down_rate))
        whole_width_lv2 = int(np.ceil(width_x / down_rate))

        patch_size = 32

        if read_patch:
            num_patch_x_lv = np.int(np.ceil(width_x / down_rate / patch_size))
            num_patch_y_lv = np.int(np.ceil(height_y / down_rate / patch_size))
            whole_width_lv = num_patch_x_lv * patch_size
            whole_height_lv = num_patch_y_lv * patch_size
            img = np.zeros((whole_height_lv,whole_width_lv, 3), dtype=np.uint8)

            for xi in range(num_patch_y_lv):
                for yi in range(num_patch_x_lv):
                    low_res_offset_x = np.int(xi * patch_size)
                    low_res_offset_y = np.int(yi * patch_size)

                    patch_start_x = start_x + np.int(low_res_offset_y * down_rate)
                    patch_start_y = start_y + np.int(low_res_offset_x * down_rate)
                    img_lv = simg.read_region((patch_start_x, patch_start_y), lv, (patch_size, patch_size))
                    img_lv = np.array(img_lv.convert('RGB'))
                    if (low_res_offset_x+patch_size) <= whole_height_lv and (low_res_offset_y+patch_size) <= whole_width_lv:
                        img[low_res_offset_x:(low_res_offset_x+patch_size), low_res_offset_y:(low_res_offset_y+patch_size), :] = img_lv

        else:
            img = simg.read_region((start_x, start_y), lv, (whole_width_lv2, whole_height_lv2))
            img = np.array(img.convert('RGB'))







        # Image.fromarray(img).resize((1000, 400)).show()
        cimg = img.copy()
        mask = np.zeros(cimg.shape, dtype=np.uint8)

        for i in range(len(layers)):
            regions = layers[i]['Regions']

            if isinstance(layers[i]['Attributes'], dict):
                clss_name = layers[i]['Attributes']['Attribute']['@Name']
            else:
                clss_name = 'unknown'

            if (len(regions) < 2):
                notFound = layers[0]
            else:
                regions = regions['Region']

                if isinstance(regions, (dict)):
                    regions = [regions]

                for j in range(len(regions)):
                    contour = regions[j]

                    vertices = contour['Vertices']['Vertex']

                    if len(vertices) <= 4:
                        continue

                    cnt = np.zeros((len(vertices), 1, 2))
                    for vi in range(len(vertices)):
                        xx = float(vertices[vi]['@X'])
                        yy = float(vertices[vi]['@Y'])
                        cnt[vi, 0, 0] = int(xx / down_rate)
                        cnt[vi, 0, 1] = int(yy / down_rate)

                    det_width = (cnt[:,0,0].max() - cnt[:,0,0].min()) * pixel_size
                    det_height = (cnt[:, 0, 1].max() - cnt[:, 0, 1].min()) * pixel_size
                    max_det = np.max((det_width, det_height))

                    if max_det > pixel_size_threshold or det_width == 0 or det_height == 0:
                        continue

                    # print('max_width = %d, %d, %d' % (max_det, det_width, det_height))

                    # draw contour
                    cv2.drawContours(cimg, [cnt.astype(int)], -1, (0, 255, 0), 1)

                    # draw masks
                    cv2.drawContours(mask, [cnt.astype(int)], -1, (255, 255, 255), -1)

    return img, cimg, mask

def get_whole_slide_good_bad(simg, layers, start_x, start_y, width_x, height_y, pixel_size, pixel_size_threshold, lv, read_patch = False):
    if 'leica.device-model' in simg.properties:
        down_rate = simg.level_downsamples[lv]
        end_x = start_x + width_x
        end_y = start_y + height_y


        max_height = width_x
        max_widths = height_y

        whole_height_lv2 = int(np.ceil((end_x - start_x) / down_rate))
        whole_width_lv2 = int(np.ceil((end_y - start_y) / down_rate))
        img = simg.read_region((start_x, start_y), lv, (whole_height_lv2, whole_width_lv2))
        img = np.array(img.convert('RGB'))
        cimg = img.copy()
        mask = np.zeros(cimg.shape, dtype=np.uint16)
        bad_mask = np.zeros(cimg.shape, dtype=np.uint16)

        good_mask_count = 1
        bad_mask_count = 1

        for i in range(len(layers)):
            regions = layers[i]['Regions']

            if isinstance(layers[i]['Attributes'], dict):
                clss_name = layers[i]['Attributes']['Attribute']['@Name']
            else:
                clss_name = 'unknown'

            if (len(regions) < 2):
                notFound = layers[0]
            else:
                regions = regions['Region']

                if isinstance(regions, (dict)):
                    regions = [regions]

                for j in range(len(regions)):
                    contour = regions[j]

                    vertices = contour['Vertices']['Vertex']

                    if len(vertices) == 2:
                        center_coordinates_x = (max_widths - ((float(vertices[0]['@X']) + float(vertices[1]['@X'])) / 2.0)) /down_rate
                        center_coordinates_y = ((float(vertices[0]['@Y']) + float(vertices[1]['@Y'])) / 2.0 /down_rate)
                        center_coordinates = (int(np.round(center_coordinates_y)), int(np.round(center_coordinates_x)))
                        radius = int(np.round(np.abs(float(vertices[0]['@X']) - float(vertices[1]['@X']))/2.0/down_rate))
                        # print('radius = %f' % radius)
                        #draw circe
                        if contour['@Text'] == 'bad':
                            # draw masks
                            cv2.circle(bad_mask, center_coordinates, radius, (bad_mask_count, bad_mask_count, bad_mask_count), -1)
                            bad_mask_count = bad_mask_count + 1
                        else:                            # draw contour
                            cv2.circle(cimg, center_coordinates, radius, (0, 255, 0), 1)
                            cv2.circle(mask, center_coordinates, radius, (good_mask_count, good_mask_count, good_mask_count), -1)
                            good_mask_count = good_mask_count+1

                    else:
                        #draw contour
                        cnt = np.zeros((len(vertices),1,2))
                        for vi in range(len(vertices)):
                            xx = float(vertices[vi]['@Y'])
                            yy = max_widths - float(vertices[vi]['@X'])
                            cnt[vi,0,0] = int(xx/down_rate)
                            cnt[vi,0,1] = int(yy/down_rate)

                        # cv2.contourArea(cnt)

                        if contour['@Text'] == 'bad':
                            # draw masks
                            cv2.circle(bad_mask, center_coordinates, radius, (bad_mask_count, bad_mask_count, bad_mask_count), -1)
                            bad_mask_count = bad_mask_count + 1
                        else:                            # draw contour
                            cv2.circle(cimg, center_coordinates, radius, (0, 255, 0), 1)
                            cv2.circle(mask, center_coordinates, radius, (good_mask_count, good_mask_count, good_mask_count), -1)
                            good_mask_count = good_mask_count+1




    elif 'aperio.Filename' in simg.properties:
        down_rate = simg.level_downsamples[lv]
        end_x = start_x + width_x
        end_y = start_y + height_y

        max_height = height_y
        max_widths = width_x

        whole_height_lv2 = int(np.ceil(height_y / down_rate))
        whole_width_lv2 = int(np.ceil(width_x / down_rate))

        patch_size = 32

        if read_patch:
            num_patch_x_lv = np.int(np.ceil(width_x / down_rate / patch_size))
            num_patch_y_lv = np.int(np.ceil(height_y / down_rate / patch_size))
            whole_width_lv = num_patch_x_lv * patch_size
            whole_height_lv = num_patch_y_lv * patch_size
            img = np.zeros((whole_height_lv,whole_width_lv, 3), dtype=np.uint8)

            for xi in range(num_patch_y_lv):
                for yi in range(num_patch_x_lv):
                    low_res_offset_x = np.int(xi * patch_size)
                    low_res_offset_y = np.int(yi * patch_size)

                    patch_start_x = start_x + np.int(low_res_offset_y * down_rate)
                    patch_start_y = start_y + np.int(low_res_offset_x * down_rate)
                    img_lv = simg.read_region((patch_start_x, patch_start_y), lv, (patch_size, patch_size))
                    img_lv = np.array(img_lv.convert('RGB'))
                    if (low_res_offset_x+patch_size) <= whole_height_lv and (low_res_offset_y+patch_size) <= whole_width_lv:
                        img[low_res_offset_x:(low_res_offset_x+patch_size), low_res_offset_y:(low_res_offset_y+patch_size), :] = img_lv

        else:
            img = simg.read_region((start_x, start_y), lv, (whole_width_lv2, whole_height_lv2))
            img = np.array(img.convert('RGB'))







        # Image.fromarray(img).resize((1000, 400)).show()
        cimg = img.copy()
        mask = np.zeros(cimg.shape, dtype=np.uint8)
        bad_mask = np.zeros(cimg.shape, dtype=np.uint8)

        for i in range(len(layers)):
            regions = layers[i]['Regions']

            if isinstance(layers[i]['Attributes'], dict):
                clss_name = layers[i]['Attributes']['Attribute']['@Name']
            else:
                clss_name = 'unknown'

            if (len(regions) < 2):
                notFound = layers[0]
            else:
                regions = regions['Region']

                if isinstance(regions, (dict)):
                    regions = [regions]

                for j in range(len(regions)):
                    contour = regions[j]

                    vertices = contour['Vertices']['Vertex']

                    if len(vertices) <= 4:
                        continue

                    if len(vertices) == 2:
                        center_coordinates_x = ((float(vertices[0]['@X']) + float(vertices[1]['@X'])) / 2.0 / down_rate)
                        center_coordinates_y = ((float(vertices[0]['@Y']) + float(vertices[1]['@Y'])) / 2.0 / down_rate)
                        center_coordinates = (int(np.round(center_coordinates_x)) , int(np.round(center_coordinates_y)))
                        radius = int(np.round(np.abs(float(vertices[0]['@X']) - float(vertices[1]['@X']))/2.0/down_rate))
                        #draw circe
                        if contour['@Text'] == 'bad':
                            # draw masks
                            cv2.circle(bad_mask, center_coordinates, radius, (bad_mask_count, bad_mask_count, bad_mask_count), -1)
                            bad_mask_count = bad_mask_count + 1
                        else:                            # draw contour
                            cv2.circle(cimg, center_coordinates, radius, (0, 255, 0), 1)
                            cv2.circle(mask, center_coordinates, radius, (good_mask_count, good_mask_count, good_mask_count), -1)
                            good_mask_count = good_mask_count+1
                    else:
                        cnt = np.zeros((len(vertices), 1, 2))
                        for vi in range(len(vertices)):
                            xx = float(vertices[vi]['@X'])
                            yy = float(vertices[vi]['@Y'])
                            cnt[vi, 0, 0] = int(xx / down_rate)
                            cnt[vi, 0, 1] = int(yy / down_rate)

                        det_width = (cnt[:,0,0].max() - cnt[:,0,0].min()) * pixel_size
                        det_height = (cnt[:, 0, 1].max() - cnt[:, 0, 1].min()) * pixel_size
                        max_det = np.max((det_width, det_height))

                        # if max_det > pixel_size_threshold or det_width == 0 or det_height == 0:
                        #     continue

                        if contour['@Text'] == 'bad':
                            # draw masks
                            cv2.circle(bad_mask, center_coordinates, radius, (bad_mask_count, bad_mask_count, bad_mask_count), -1)
                            bad_mask_count = bad_mask_count + 1
                        else:                            # draw contour
                            cv2.circle(cimg, center_coordinates, radius, (0, 255, 0), 1)
                            cv2.circle(mask, center_coordinates, radius, (good_mask_count, good_mask_count, good_mask_count), -1)
                            good_mask_count = good_mask_count+1
                        # print('max_width = %d, %d, %d' % (max_det, det_width, det_height))

                        # # draw contour
                        # cv2.drawContours(cimg, [cnt.astype(int)], -1, (0, 255, 0), 1)
                        #
                        # # draw masks
                        # cv2.drawContours(mask, [cnt.astype(int)], -1, (255, 255, 255), -1)

    return img, cimg, mask, bad_mask


def get_whole_slide(simg, contours, start_x, start_y, end_x, end_y, create_number_per_image, img_size):

    lv = 2
    down_rate = simg.level_downsamples[lv]

    try:
        max_height = int(simg.properties['aperio.OriginalWidth'])
        max_widths = int(simg.properties['aperio.OriginalHeight'])
    except:
        max_height = int(simg.properties['openslide.bounds-width'])
        max_widths = int(simg.properties['openslide.bounds-height'])




    whole_height_lv2 = int(np.ceil((end_x - start_x) / down_rate))
    whole_width_lv2 = int(np.ceil((end_y - start_y) / down_rate))
    img = simg.read_region((start_x, start_y), 2, (whole_height_lv2, whole_width_lv2))
    img = np.array(img.convert('RGB'))
    cimg = img.copy()
    mask = np.zeros(cimg.shape, dtype=np.uint8)

    for ci in range(len(contours)):
        contour = contours[ci]
        vertices = contour['Vertices']['Vertex']

        cnt = np.zeros((len(vertices),1,2))
        for vi in range(len(vertices)):
            xx = float(vertices[vi]['@Y'])
            yy = max_widths - float(vertices[vi]['@X'])
            cnt[vi,0,0] = int(xx/down_rate)
            cnt[vi,0,1] = int(yy/down_rate)



        #draw contour
        cv2.drawContours(cimg, [cnt.astype(int)], -1, (0, 255, 0), 1)

        #draw masks
        cv2.drawContours(mask, [cnt.astype(int)], -1, (255, 255, 255), -1)

    # Image.fromarray(cimg).show()


    return img, cimg, mask


def get_MASK(simg, region, contour):
    vertices = region['Vertices']['Vertex']
    x0 = float(vertices[0]['@X'])
    y0 = float(vertices[0]['@Y'])
    z0 = float(vertices[0]['@Z'])
    x1 = float(vertices[1]['@X'])
    y1 = float(vertices[1]['@Y'])
    z1 = float(vertices[1]['@Z'])
    x2 = float(vertices[2]['@X'])
    y2 = float(vertices[2]['@Y'])
    z2 = float(vertices[2]['@Z'])
    x3 = float(vertices[3]['@X'])
    y3 = float(vertices[3]['@Y'])
    z3 = float(vertices[3]['@Z'])

    # get manual ROI coordinate
    ys = int(round(x0))
    xs = int(round(y0))
    yss = int(round(x1))
    xss = int(round(y2))
    widths = int(round(x1 - x0))
    heights = int(round(y2 - y1))

    start_x, start_y = get_nonblack_starting_point(simg)
    #isimg.read_region((44600, 82700), 0, (widths,heights).show()
    max_height = int(simg.properties['openslide.bounds-height'])
    img = simg.read_region((start_x+xs, max_height-yss+start_y), 0, (heights,widths))
    img = np.array(img.convert('RGB'))

    cimg = img.copy()
    vertices = contour['Vertices']['Vertex']
    cnt = np.zeros((len(vertices),1,2))
    for vi in range(len(vertices)):
        xx = float(vertices[vi]['@Y'])-xs
        yy = yss - float(vertices[vi]['@X'])
        cnt[vi,0,0] = int(xx)
        cnt[vi,0,1] = int(yy)

    cv2.drawContours(cimg, [cnt.astype(int)], -1, (0, 255, 0), 3)

    #draw mask
    mask = np.zeros(cimg.shape, dtype=np.uint8)
    cv2.drawContours(mask, [cnt.astype(int)], -1, (255, 255, 255), -1)

    # Image.fromarray(cimg).show()


    return img, cimg, mask


def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return ymin,ymax+1,xmin,xmax+1
    # return img[ymin:ymax+1, xmin:xmax+1]


def save_cropped_img_mask(big_img_file, big_mask_file, output_dir, create_number_per_image, img_size, fname):
    #check if finish all files
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

    #get ROI list
    big_mask = Image.open(big_mask_file)
    big_mask_binary = np.array(big_mask.convert('L'))

    big_img = Image.open(big_img_file)
    big_img_arr = np.array(big_img)

    labels = label(big_mask_binary)
    bincounts = np.bincount(labels.flat)
    sorted_ind = np.argsort(-bincounts)
    # Image.fromarray(labels.astype('uint8')*10).show()

    h,w,d = big_img_arr.shape

    for oi in range(1, len(sorted_ind)):
        seg_id = sorted_ind[oi]
        ROI = labels == seg_id
        bbox = bbox2(ROI.astype(np.uint8))
        # big_img_arr[bbox[0]:bbox[1],bbox[2]:bbox[3],:]
        # Image.fromarray(big_img_arr[bbox[0]:bbox[1], bbox[2]:bbox[3], :]).show()
        box_h = bbox[1] - bbox[0]
        box_w = bbox[3] - bbox[2]
        x0 = bbox[0]
        y0 = bbox[2]

        rand_maxx = img_size[0] - box_h - 1
        rand_mayy = img_size[1] - box_w - 1

        # x0_rand = x0
        # y0_rand = y0
        for ri in range(create_number_per_image):
            x0_rand = x0 - int(random.randint(0, rand_maxx))
            y0_rand = y0 - int(random.randint(0, rand_mayy))

            png_name = '%s-x-contour%03d-x-%d-x-%d.png' % (fname, oi, x0_rand, y0_rand)
            output_img_file = os.path.join(output_QA_dir, png_name)
            img_out_file = os.path.join(output_img_dir, png_name)
            mask_out_file = os.path.join(output_mask_dir, png_name)

            if os.path.exists(output_img_file) and os.path.exists(img_out_file) and os.path.exists(mask_out_file):
                continue

            x0_rand = min(x0_rand, h - img_size[0] - 1)
            x0_rand = max(x0_rand, 0)
            y0_rand = min(y0_rand, w - img_size[1] - 1)
            y0_rand = max(y0_rand, 0)

            img = big_img_arr[x0_rand:x0_rand+img_size[0],y0_rand:y0_rand+img_size[1],:]
            mask = big_mask_binary[x0_rand:x0_rand+img_size[0],y0_rand:y0_rand+img_size[1]]

            imgI = Image.fromarray(img).convert('RGB')
            maskI = Image.fromarray(mask).convert('RGB')
            overlayI = Image.blend(imgI, maskI, 0.5)
            # new_img.save(output_img_file)



            overlayI.save(output_img_file)
            imgI.save(img_out_file)
            maskI.save(mask_out_file)

            roi_count = roi_count + 1

    return roi_count

def save_cropped_img_mask_good_bad(big_img_arr, big_mask_binary, output_dir, create_number_per_image, img_size, fname, type='good'):
    #check if finish all files
    if type == 'good':
        output_QA_dir = os.path.join(output_dir,'QA_roi')
        output_img_dir = os.path.join(output_dir,'train_roi')
        output_mask_dir = os.path.join(output_dir,'labels_roi')
    else:
        output_QA_dir = os.path.join(output_dir, 'bad_QA_roi')
        output_img_dir = os.path.join(output_dir, 'bad_train_roi')
        output_mask_dir = os.path.join(output_dir, 'bad_labels_roi')

    random.seed(0)

    if not os.path.exists(output_QA_dir):
        os.makedirs(output_QA_dir)
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
    if not os.path.exists(output_mask_dir):
        os.makedirs(output_mask_dir)

    roi_count = 0

    #get ROI list
    # big_mask = Image.open(big_mask_file)
    # big_mask_binary = np.array(big_mask.convert('L'))

    # big_img = Image.open(big_img_file)
    # big_img_arr = np.array(big_img)
    unique_labels = np.unique(big_mask_binary)
    sorted_ind = np.argsort(unique_labels)
    # Image.fromarray(labels.astype('uint8')*10).show()

    h,w,d = big_img_arr.shape

    for oi in range(1, len(sorted_ind)):
        seg_id = sorted_ind[oi]
        ROI = big_mask_binary == seg_id
        try:
            bbox = bbox2(ROI.astype(np.uint8))
        except:
            continue
        # big_img_arr[bbox[0]:bbox[1],bbox[2]:bbox[3],:]
        # Image.fromarray(big_img_arr[bbox[0]:bbox[1], bbox[2]:bbox[3], :]).show()
        box_h = bbox[1] - bbox[0]
        box_w = bbox[3] - bbox[2]
        x0 = bbox[0]
        y0 = bbox[2]

        rand_maxx = img_size[0] - box_h - 1
        rand_mayy = img_size[1] - box_w - 1

        # x0_rand = x0
        # y0_rand = y0
        for ri in range(create_number_per_image):
            x0_rand = x0 - int(random.randint(0, rand_maxx))
            y0_rand = y0 - int(random.randint(0, rand_mayy))

            png_name = '%s-x-contour%03d-x-%d-x-%d.png' % (fname, oi, x0_rand, y0_rand)
            output_img_file = os.path.join(output_QA_dir, png_name)
            img_out_file = os.path.join(output_img_dir, png_name)
            mask_out_file = os.path.join(output_mask_dir, png_name)
            tiff_name = '%s-x-contour%03d-x-%d-x-%d.tiff' % (fname, oi, x0_rand, y0_rand)
            tiff_mask_out_file = os.path.join(output_mask_dir, tiff_name)

            if os.path.exists(output_img_file) and os.path.exists(img_out_file) and os.path.exists(mask_out_file) and os.path.exists(tiff_mask_out_file):
                continue

            x0_rand = min(x0_rand, h - img_size[0] - 1)
            x0_rand = max(x0_rand, 0)
            y0_rand = min(y0_rand, w - img_size[1] - 1)
            y0_rand = max(y0_rand, 0)

            img = big_img_arr[x0_rand:x0_rand+img_size[0],y0_rand:y0_rand+img_size[1],:]
            mask = big_mask_binary[x0_rand:x0_rand+img_size[0],y0_rand:y0_rand+img_size[1]]

            tiff = TIFF.open(tiff_mask_out_file, mode='w')
            tiff.write_image(mask[:,:,0])
            tiff.close()

            imgI = Image.fromarray(img).convert('RGB')
            maskI = Image.fromarray(mask.astype(np.uint8)).convert('RGB')
            overlayI = Image.blend(imgI, maskI, 0.5)
            # new_img.save(output_img_file)

            # tiff = TIFF.open(tiff_mask_out_file, mode='r')
            # ar = tiff.read_image()
            # tiff.close()

            overlayI.save(output_img_file)
            imgI.save(img_out_file)
            maskI.save(mask_out_file)

            roi_count = roi_count + 1

    return roi_count

def save_cropped_img_mask_atubular(big_img_arr, big_mask_binary, output_dir, create_number_per_image, img_size, fname):
    #check if finish all files
    output_QA_dir = os.path.join(output_dir,'QA_roi')
    output_img_dir = os.path.join(output_dir,'train_roi')
    output_mask_dir = os.path.join(output_dir,'labels_roi')

    random.seed(0)

    if not os.path.exists(output_QA_dir):
        os.makedirs(output_QA_dir)
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
    if not os.path.exists(output_mask_dir):
        os.makedirs(output_mask_dir)

    roi_count = 0

    #get ROI list
    # big_mask = Image.open(big_mask_file)
    # big_mask_binary = np.array(big_mask.convert('L'))

    # big_img = Image.open(big_img_file)
    # big_img_arr = np.array(big_img)
    big_mask_binary = np.array(Image.fromarray(big_mask_binary).convert('L'))

    labels = label(big_mask_binary)
    bincounts = np.bincount(labels.flat)
    sorted_ind = np.argsort(-bincounts)
    # Image.fromarray(labels.astype('uint8')*10).show()

    h,w,d = big_img_arr.shape

    for oi in range(1, len(sorted_ind)):
        seg_id = sorted_ind[oi]
        ROI = labels == seg_id
        ROI = ROI * 1
        bbox = bbox2(ROI)
        # big_img_arr[bbox[0]:bbox[1],bbox[2]:bbox[3],:]
        # Image.fromarray(big_img_arr[bbox[0]:bbox[1], bbox[2]:bbox[3], :]).show()
        box_h = bbox[1] - bbox[0]
        box_w = bbox[3] - bbox[2]
        x0 = bbox[0]
        y0 = bbox[2]

        rand_maxx = img_size[0] - box_h - 1
        rand_mayy = img_size[1] - box_w - 1

        # x0_rand = x0
        # y0_rand = y0
        for ri in range(create_number_per_image):
            x0_rand = x0 - int(random.randint(0, rand_maxx))
            y0_rand = y0 - int(random.randint(0, rand_mayy))

            png_name = '%s-x-contour%03d-x-%d-x-%d.png' % (fname, oi, x0_rand, y0_rand)
            output_img_file = os.path.join(output_QA_dir, png_name)
            img_out_file = os.path.join(output_img_dir, png_name)
            mask_out_file = os.path.join(output_mask_dir, png_name)

            if os.path.exists(output_img_file) and os.path.exists(img_out_file) and os.path.exists(mask_out_file):
                continue

            x0_rand = min(x0_rand, h - img_size[0] - 1)
            x0_rand = max(x0_rand, 0)
            y0_rand = min(y0_rand, w - img_size[1] - 1)
            y0_rand = max(y0_rand, 0)

            img = big_img_arr[x0_rand:x0_rand+img_size[0],y0_rand:y0_rand+img_size[1],:]
            mask = big_mask_binary[x0_rand:x0_rand+img_size[0],y0_rand:y0_rand+img_size[1]]

            imgI = Image.fromarray(img).convert('RGB')
            maskI = Image.fromarray(mask).convert('RGB')
            overlayI = Image.blend(imgI, maskI, 0.5)
            # new_img.save(output_img_file)



            overlayI.save(output_img_file)
            imgI.save(img_out_file)
            maskI.save(mask_out_file)

            roi_count = roi_count + 1

    return roi_count


def save_cropped_img_mask_bad(big_img_arr, big_mask_binary, output_dir, create_number_per_image, img_size, fname):
    #check if finish all files
    output_QA_dir = os.path.join(output_dir,'bad_QA_roi')
    output_img_dir = os.path.join(output_dir,'bad_train_roi')
    output_mask_dir = os.path.join(output_dir,'bad_labels_roi')

    random.seed(0)

    if not os.path.exists(output_QA_dir):
        os.makedirs(output_QA_dir)
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
    if not os.path.exists(output_mask_dir):
        os.makedirs(output_mask_dir)

    roi_count = 0

    #get ROI list
    # big_mask = Image.open(big_mask_file)
    # big_mask_binary = np.array(big_mask.convert('L'))

    # big_img = Image.open(big_img_file)
    # big_img_arr = np.array(big_img)
    big_mask_binary = np.array(Image.fromarray(big_mask_binary).convert('L'))

    labels = label(big_mask_binary)
    bincounts = np.bincount(labels.flat)
    sorted_ind = np.argsort(-bincounts)
    # Image.fromarray(labels.astype('uint8')*10).show()

    h,w,d = big_img_arr.shape

    for oi in range(1, len(sorted_ind)):
        seg_id = sorted_ind[oi]
        ROI = labels == seg_id
        ROI = ROI * 1
        bbox = bbox2(ROI)
        # big_img_arr[bbox[0]:bbox[1],bbox[2]:bbox[3],:]
        # Image.fromarray(big_img_arr[bbox[0]:bbox[1], bbox[2]:bbox[3], :]).show()
        box_h = bbox[1] - bbox[0]
        box_w = bbox[3] - bbox[2]
        x0 = bbox[0]
        y0 = bbox[2]

        rand_maxx = img_size[0] - box_h - 1
        rand_mayy = img_size[1] - box_w - 1

        # x0_rand = x0
        # y0_rand = y0
        for ri in range(create_number_per_image):
            x0_rand = x0 - int(random.randint(0, rand_maxx))
            y0_rand = y0 - int(random.randint(0, rand_mayy))

            png_name = '%s-x-contour%03d-x-%d-x-%d.png' % (fname, oi, x0_rand, y0_rand)
            output_img_file = os.path.join(output_QA_dir, png_name)
            img_out_file = os.path.join(output_img_dir, png_name)
            mask_out_file = os.path.join(output_mask_dir, png_name)

            if os.path.exists(output_img_file) and os.path.exists(img_out_file) and os.path.exists(mask_out_file):
                continue

            x0_rand = min(x0_rand, h - img_size[0] - 1)
            x0_rand = max(x0_rand, 0)
            y0_rand = min(y0_rand, w - img_size[1] - 1)
            y0_rand = max(y0_rand, 0)

            img = big_img_arr[x0_rand:x0_rand+img_size[0],y0_rand:y0_rand+img_size[1],:]
            mask = big_mask_binary[x0_rand:x0_rand+img_size[0],y0_rand:y0_rand+img_size[1]]

            imgI = Image.fromarray(img).convert('RGB')
            maskI = Image.fromarray(mask).convert('RGB')
            overlayI = Image.blend(imgI, maskI, 0.5)
            # new_img.save(output_img_file)



            overlayI.save(output_img_file)
            imgI.save(img_out_file)
            maskI.save(mask_out_file)

            roi_count = roi_count + 1

    return roi_count


