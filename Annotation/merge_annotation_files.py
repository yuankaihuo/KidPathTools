import openslide
import xmltodict
import numpy as np
from PIL import Image
import os
import cv2
import glob

def read_xml(xml_file):
    # read region
    with open(xml_file) as fd:
        doc = xmltodict.parse(fd.read())
    layers = doc['Annotations']['Annotation']

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

            # for j in range(len(regions)):
            #     contour = regions[j]
            #     vertices = contour['Vertices']['Vertex']

    return regions

def if_point_in_mask(center_coordinates, cnt_seg):
    x_min = cnt_seg[:, 0, 0].min()
    x_max = cnt_seg[:, 0, 0].max()
    y_min = cnt_seg[:, 0, 1].min()
    y_max = cnt_seg[:, 0, 1].max()

    x0 = center_coordinates[0]
    y0 = center_coordinates[1]
    if x0 > x_min and x0 < x_max and y0 > y_min and y0 < y_max:
        return True
    else:
        return False


def merge_raw_seg(regions_raw, regions_seg):
    for j in range(len(regions_raw)):
        contour_raw = regions_raw[j]
        vertices_raw = contour_raw['Vertices']['Vertex']

        center_coordinates_x = ((float(vertices_raw[0]['@X']) + float(vertices_raw[1]['@X'])) / 2.0)
        center_coordinates_y = ((float(vertices_raw[0]['@Y']) + float(vertices_raw[1]['@Y'])) / 2.0)
        center_coordinates = (center_coordinates_x, center_coordinates_y)
        radius = np.abs(float(vertices_raw[0]['@X']) - float(vertices_raw[1]['@X'])) / 2.0

        match_ind = []
        for i in range(len(regions_seg)):
            contour_seg = regions_seg[i]
            vertices_seg = contour_seg['Vertices']['Vertex']

            cnt_seg = np.zeros((len(vertices_seg), 1, 2))
            for vi in range(len(vertices_seg)):
                xx = float(vertices_seg[vi]['@X'])
                yy = float(vertices_seg[vi]['@Y'])
                cnt_seg[vi, 0, 0] = int(xx)
                cnt_seg[vi, 0, 1] = int(yy)

            if if_point_in_mask(center_coordinates, cnt_seg):
                match_ind.append(i)

        if match_ind == []:
            contour_raw['@Text'] = 'miss'
        else:
            assert(len(match_ind)) == 1
            match_ind = match_ind[0]
            id = regions_raw[j]['@Id']
            display_id = regions_raw[j]['@DisplayId']
            regions_raw[j] = regions_seg[match_ind]
            regions_raw[j]['@Id'] = id
            regions_raw[j]['@DisplayId'] = display_id
        print('%d/%d' % (j ,len(regions_raw)))
    return regions_raw

def write_xml_file(input_raw_file, regions_merge, output_file):
    # read region
    with open(input_raw_file) as fd:
        doc = xmltodict.parse(fd.read())
    layers = doc['Annotations']['Annotation']

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
            regions['Region'] = regions_merge


    out = xmltodict.unparse(doc, pretty=True)
    with open(output_file, 'wb') as file:
        file.write(out.encode('utf-8'))

if __name__ == "__main__":

    # subname = '23499_2017-04-07 20_15_04.xml'
    subname = '23681_2017-04-07 20_27_42.xml'

    input_raw_file = os.path.join('/media/huoy1/48EAE4F7EAE4E264/Projects/fromGe/raw',subname)
    input_seg_file = os.path.join('/media/huoy1/48EAE4F7EAE4E264/Projects/fromGe/seg',subname)
    output_file = os.path.join('/media/huoy1/48EAE4F7EAE4E264/Projects/fromGe/output',subname)

    regions_raw = read_xml(input_raw_file)
    regions_seg = read_xml(input_seg_file)

    regions_merge = merge_raw_seg(regions_raw, regions_seg)
    write_xml_file(input_raw_file, regions_merge, output_file)