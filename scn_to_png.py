import openslide
import xmltodict
import numpy as np
from PIL import Image
import os
import cv2
import random
from skimage.measure import label

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








