#!/usr/bin/env python3

import os
import re
import datetime
import numpy as np
from itertools import groupby
from skimage import measure
from PIL import Image
from pycocotools import mask
from skimage.transform import resize

convert = lambda text: int(text) if text.isdigit() else text.lower()
natrual_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]

def resize_binary_mask(array, new_size):
    image = Image.fromarray(array.astype(np.uint8)*255)
    image = image.resize(new_size)
    return np.asarray(image).astype(np.bool_)

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
                counts.append(0)
        counts.append(len(list(elements)))

    return rle

def binary_mask_to_polygon(binary_vol_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation

    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.

    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    for slice in range(binary_vol_mask.shape[2]):
        binary_mask = binary_vol_mask[:,:,slice]
        padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
        contours = measure.find_contours(padded_binary_mask, 0.5)
        contours = np.subtract(contours, 1)
        for contour in contours:
            contour = close_contour(contour)
            contour = measure.approximate_polygon(contour, tolerance)
            if len(contour) < 3:
                continue
            contour = np.flip(contour, axis=1)
            contour_3d = np.zeros((contour.shape[0],contour.shape[1]+1))
            for ci in range(len(contour_3d)):
                contour_3d[ci][0] = contour[ci][0]
                contour_3d[ci][1] = contour[ci][1]
                contour_3d[ci][2] = slice
            segmentation = contour_3d.ravel().tolist()
            # after padding and subtracting 1 we may get -0.5 points in our segmentation
            segmentation = [0 if i < 0 else i for i in segmentation]
            polygons.append(segmentation)

    return polygons

def create_image_info(image_id, file_name, phases, image_size,
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1, coco_url="", flickr_url=""):

    image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": image_size[0],
            "height": image_size[1],
            'depth': image_size[2],
            "date_captured": date_captured,
            "license": license_id,
    }

    for pi in range(len(phases)):
        phase = phases[pi]
        phase_image_name = '%s-x-%s.nii.gz' % (file_name, phase)
        image_info[phase] = phase_image_name


    return image_info

def bbox1(img):
    a = np.where(img != 0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    return bbox

def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

def bbox2_3D(img):

    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return rmin, rmax, cmin, cmax, zmin, zmax

def create_annotation_info(annotation_id, image_id, category_info, binary_mask,
                           image_size=None, tolerance=2, bounding_box=None):

    if image_size is not None:
        binary_mask = resize(binary_mask, image_size, order=0, anti_aliasing=False, preserve_range=True)

    binary_mask_encoded = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))

    area = mask.area(binary_mask_encoded)
    # area = area.sum() #from 2d to 3d
    if area.sum() < 1:
        return None

    if bounding_box is None:
        bounding_box = mask.toBbox(binary_mask_encoded)
        # for 3D
        rmin, rmax, cmin, cmax, zmin, zmax = bbox2_3D(binary_mask)
        bounding_box3 = np.zeros(6)
        bounding_box3[0] = cmin
        bounding_box3[1] = rmin
        bounding_box3[2] = zmin
        bounding_box3[3] = cmax - cmin + 1
        bounding_box3[4] = rmax - rmin + 1
        bounding_box3[5] = zmax - zmin + 1

        # for 2D
        # rmin, rmax, cmin, cmax = bbox2(binary_mask)
        # bounding_box2 = np.zeros(4)
        # bounding_box2[0] = cmin
        # bounding_box2[1] = rmin
        # bounding_box2[2] = cmax - cmin + 1
        # bounding_box2[3] = rmax - rmin + 1


    if category_info["is_crowd"]:
        is_crowd = 1
        segmentation = binary_mask_to_rle(binary_mask)
    else :
        is_crowd = 0
        segmentation = binary_mask_to_polygon(binary_mask, tolerance)
        if not segmentation:
            return None

    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_info["id"],
        "iscrowd": is_crowd,
        "area": area.tolist(),
        "bbox2d": bounding_box.tolist(),
        "bbox": bounding_box3.tolist(),
        "segmentation": segmentation,
        "depth": binary_mask.shape[2],
        "width": binary_mask.shape[1],
        "height": binary_mask.shape[0],
    }

    return annotation_info
