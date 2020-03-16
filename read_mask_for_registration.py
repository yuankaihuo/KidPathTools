import openslide
import xmltodict
import numpy as np
from PIL import Image
import os
import cv2
import pandas as pd

def big_contain_small(big_box, small_box):
    if big_box[0] < small_box[0] and big_box[1] < small_box[1] and big_box[0]+big_box[2] > small_box[0]+small_box[2] and big_box[1]+big_box[3] > small_box[1]+small_box[3]:
        return True
    else:
        return False

def get_bbox_from_mask(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def read_annotations(simg, xml_file, output_dir, auto_xml_file = None):
    new_height = 4000


    # read region
    with open(xml_file) as fd:
        doc = xmltodict.parse(fd.read())

    try:
        layer1 = doc['Annotations']['Annotation'][0]
        layer2 = doc['Annotations']['Annotation'][1]
        # two layers
        contours = layer1['Regions']['Region']
        contours2 = layer2['Regions']['Region']
    except:
        if os.path.basename(xml_file) == '14-50.xml':
            contours = doc['Annotations']['Annotation']['Regions']['Region'][0:14]
            contours2 = doc['Annotations']['Annotation']['Regions']['Region'][14:-1]

    #auto boxes
    if not auto_xml_file is None:
        with open(auto_xml_file) as fd:
            doc2 = xmltodict.parse(fd.read())
            layer2 = doc2['Annotations']['Annotation']
            contours2 = layer2['Regions']['Region']

    start_x = 0
    start_y = 0

    bboxs_big = []
    for i in range(len(contours)):
        contour = contours[i]
        bbox = get_bbox(simg, contour, start_x, start_y)
        bboxs_big.append(bbox)

    # img, cimg, mask, bbox = get_contour(simg, contours[2], start_x, start_y)
    # img_1 = Image.fromarray(img)
    # new_width = int(new_height * img_1.width / img_1.height)


    bboxs_small = []
    for j in range(len(contours2)):
        contour2 = contours2[j]
        bbox2 = get_bbox(simg, contour2, start_x, start_y)
        bboxs_small.append(bbox2)

    bbox_dicts = []
    for bi in range(len(bboxs_big)):
        bbox_dict = {}
        bbox_dict['big'] = bboxs_big[bi]
        bbox_dict['small'] = None
        bbox_dict['coordinate'] = None
        bbox_dict['reshape_coordinate'] = None
        big_box = bboxs_big[bi]
        found_small_count = 0
        for bj in range(len(bboxs_small)):
            small_box = bboxs_small[bj]
            if big_contain_small(big_box, small_box):
                bbox_dict['small'] = small_box

                numpy_out_file = os.path.join(output_dir, '%s-x-ROI_%d-x-BOX_%d.npy' %
                                              (os.path.basename(xml_file).replace('.xml', ''), bi, found_small_count))
                if os.path.exists(numpy_out_file):
                    found_small_count = found_small_count + 1
                    continue

                relative_box = np.zeros(4)
                relative_box[0] = small_box[0] - big_box[0]
                relative_box[1] = small_box[1] - big_box[1]
                relative_box[2] = small_box[2]
                relative_box[3] = small_box[3]


                img = np.zeros((big_box[3], big_box[2], 3), dtype=np.uint8)
                relative_box = relative_box.astype(np.int)
                img[relative_box[1]:(relative_box[1]+relative_box[3]),relative_box[0]:(relative_box[0]+relative_box[2]),:] = 255


                # img3 = simg.read_region((big_box[0], big_box[1]), 0, (big_box[2], big_box[3]))
                # # # simg.read_region((read_x0, read_y0), 0, (read_height, read_widths)).resize([100,100],Image.ANTIALIAS).show()
                # # # img = simg.read_region((10000, 36000), 0, (5000,5000)).resize([100,100],Image.ANTIALIAS).show()
                # relative_box = relative_box.astype(np.int)
                # img3 = np.array(img3.convert('RGB'))
                # img4 = img3[relative_box[1]:(relative_box[1]+relative_box[3]),relative_box[0]:(relative_box[0]+relative_box[2]),:]

                img_1 = Image.fromarray(img)

                new_width = int(new_height * img_1.width / img_1.height)
                img_all_out = img_1.resize((new_width, new_height), Image.NEAREST)
                rmin, rmax, cmin, cmax = get_bbox_from_mask(np.array(img_all_out))
                resize_box = [cmin, rmin, (cmax-cmin), (rmax-rmin)]

                img_all_out_file = os.path.join(output_dir, '%s-x-ROI_%d-x-BOX_%d-x-%d-x-%d-x-%d-x-%d.jpg' %
                                                (os.path.basename(xml_file).replace('.xml', ''), bi, found_small_count,
                                                 resize_box[0], resize_box[1], resize_box[2], resize_box[3]))
                img_all_out.save(img_all_out_file)


                np.save(numpy_out_file, img)



                # bbox_dict['coordinate'] =
                found_small_count = found_small_count+1
                # break
        # bbox_dicts.append(bbox_dict)
        # assert found_small_count<2


    # bboxs_big
    # bboxs_small


def read_mask(simg,xml_file,output_dir):

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

    # start_x, start_y = get_nonblack_starting_point(simg)
    start_x = 0
    start_y = 0

    try:
        contours['Vertices']
        img, cimg, mask, bbox = get_contour(simg, contours, start_x, start_y)
        img_1 = np.concatenate((img, img, img, img), axis=1)
        img_all = np.concatenate((img_1, img_1), axis=0)
        img_all_out = Image.fromarray(img_all)
        img_all_out_file = os.path.join(output_dir, '%s-x-ROI_%d-x-%d-x-%d-x-%d-x-%d.jpg' %
                                        (os.path.basename(xml_file).replace('.xml',''),0,
                                         bbox[0], bbox[1], bbox[2], bbox[3]))
        img_all_out.save(img_all_out_file)
    except:
        for i in range(len(contours)):
            contour = contours[i]
            img, cimg, mask, bbox = get_contour(simg,  contour, start_x, start_y)
            # Image.fromarray(cimg).resize((800, 800), Image.ANTIALIAS).show()

            # img_1 = np.concatenate((img, img, img, img), axis=1)
            # img_all = np.concatenate((img_1, img_1), axis=0)
            img_1 = Image.fromarray(img)
            new_height = 4000
            new_width = int(new_height * img_1.width / img_1.height)
            img_all_out = img_1.resize((new_width, new_height), Image.ANTIALIAS)

            img_all_out_file = os.path.join(output_dir, '%s-x-ROI_%d-x-%d-x-%d-x-%d-x-%d.jpg' %
                                        (os.path.basename(xml_file).replace('.xml',''),i,
                                         bbox[0], bbox[1], bbox[2], bbox[3]))
            img_all_out.save(img_all_out_file)


            numpy_out_file = os.path.join(output_dir, '%s-x-ROI_%d.npy' %
                                        (os.path.basename(xml_file).replace('.xml',''),i))
            np.save(numpy_out_file, img)

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
    # max_widths = int(simg.properties['openslide.bounds-height'])
    max_widths = int(simg.properties['aperio.OriginalHeight'])
    img = simg.read_region((start_x+xs, max_widths-yss+start_y), 0, (heights,widths))
    img = np.array(img.convert('RGB'))
    return img

def get_contour(simg, contour, start_x, start_y):
    max_height = int(simg.properties['aperio.OriginalWidth'])
    max_widths = int(simg.properties['aperio.OriginalHeight'])
    vertices = contour['Vertices']['Vertex']
    x_min = max_height
    x_max = 0
    y_min = max_widths
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
        xx_min = max(x_min-50, 0)
        xx_max = min(x_max+50, max_height)
        yy_min = max(y_min-50, 0)
        yy_max = min(y_max+50, max_widths)

        heights = xx_max-xx_min
        widths = yy_max-yy_min

        xs = xx_min
        yss = yy_min


    # cnt = np.zeros((len(vertices),1,2))
    # for vi in range(len(vertices)):
    #     xx = float(vertices[vi]['@Y'])
    #     yy = float(vertices[vi]['@X'])
    #     cnt[vi,0,0] = int(xx)
    #     cnt[vi,0,1] = int(yy)



    #isimg.read_region((44600, 82700), 0, (widths,heights).show()

    read_x0 = xs
    read_y0 = yss
    read_height = heights
    read_widths = widths
    bbox = (read_x0, read_y0, read_height, read_widths)

    img = simg.read_region((read_x0, read_y0), 0, (read_height,read_widths))
    # simg.read_region((read_x0, read_y0), 0, (read_height, read_widths)).resize([100,100],Image.ANTIALIAS).show()
    # img = simg.read_region((10000, 36000), 0, (5000,5000)).resize([100,100],Image.ANTIALIAS).show()

    img = np.array(img.convert('RGB'))

    cimg = img.copy()
    vertices = contour['Vertices']['Vertex']
    cnt = np.zeros((len(vertices),1,2))
    for vi in range(len(vertices)):
        xx = float(vertices[vi]['@X'])-xs
        yy = float(vertices[vi]['@Y'])-yss

        # xx = float(vertices[vi]['@Y'])-xs
        # yy = yss - float(vertices[vi]['@X'])
        cnt[vi,0,0] = int(xx)
        cnt[vi,0,1] = int(yy)

    cv2.drawContours(cimg, [cnt.astype(int)], -1, (0, 255, 0), 3)

    #draw mask
    mask = np.zeros(cimg.shape, dtype=np.uint8)
    cv2.drawContours(mask, [cnt.astype(int)], -1, (255, 255, 255), -1)

    # Image.fromarray(cimg).show()


    return img, cimg, mask, bbox

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



def get_bbox(simg, contour, start_x, start_y):
    max_height = int(simg.properties['aperio.OriginalWidth'])
    max_widths = int(simg.properties['aperio.OriginalHeight'])
    vertices = contour['Vertices']['Vertex']
    x_min = max_height
    x_max = 0
    y_min = max_widths
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
        xx_min = max(x_min-50, 0)
        xx_max = min(x_max+50, max_height)
        yy_min = max(y_min-50, 0)
        yy_max = min(y_max+50, max_widths)

        heights = xx_max-xx_min
        widths = yy_max-yy_min

        xs = xx_min
        yss = yy_min


    # cnt = np.zeros((len(vertices),1,2))
    # for vi in range(len(vertices)):
    #     xx = float(vertices[vi]['@Y'])
    #     yy = float(vertices[vi]['@X'])
    #     cnt[vi,0,0] = int(xx)
    #     cnt[vi,0,1] = int(yy)



    #isimg.read_region((44600, 82700), 0, (widths,heights).show()

    read_x0 = xs
    read_y0 = yss
    read_height = heights
    read_widths = widths
    bbox = (read_x0, read_y0, read_height, read_widths)

    return bbox