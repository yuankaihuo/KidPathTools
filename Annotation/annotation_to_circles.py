import openslide
import xmltodict
import numpy as np
from PIL import Image
import os
import cv2
import glob
from math import sqrt
from collections import OrderedDict
#
# Smallest enclosing circle
#
# Copyright (c) 2014 Project Nayuki
# https://www.nayuki.io/page/smallest-enclosing-circle
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program (see COPYING.txt).
# If not, see <http://www.gnu.org/licenses/>.
#

import math, random


# Data conventions: A point is a pair of floats (x, y). A circle is a triple of floats (center x, center y, radius).

# Returns the smallest circle that encloses all the given points. Runs in expected O(n) time, randomized.
# Input: A sequence of pairs of floats or ints, e.g. [(0,5), (3.1,-2.7)].
# Output: A triple of floats representing a circle.
# Note: If 0 points are given, None is returned. If 1 point is given, a circle of radius 0 is returned.
#
# Initially: No boundary points known
def make_circle(points):
    # Convert to float and randomize order
    shuffled = [(float(x), float(y)) for (x, y) in points]
    random.shuffle(shuffled)

    # Progressively add points to circle or recompute circle
    c = None
    for (i, p) in enumerate(shuffled):
        if c is None or not is_in_circle(c, p):
            c = _make_circle_one_point(shuffled[: i + 1], p)
    return c


# One boundary point known
def _make_circle_one_point(points, p):
    c = (p[0], p[1], 0.0)
    for (i, q) in enumerate(points):
        if not is_in_circle(c, q):
            if c[2] == 0.0:
                c = make_diameter(p, q)
            else:
                c = _make_circle_two_points(points[: i + 1], p, q)
    return c


# Two boundary points known
def _make_circle_two_points(points, p, q):
    circ = make_diameter(p, q)
    left = None
    right = None
    px, py = p
    qx, qy = q

    # For each point not in the two-point circle
    for r in points:
        if is_in_circle(circ, r):
            continue

        # Form a circumcircle and classify it on left or right side
        cross = _cross_product(px, py, qx, qy, r[0], r[1])
        c = make_circumcircle(p, q, r)
        if c is None:
            continue
        elif cross > 0.0 and (
                left is None or _cross_product(px, py, qx, qy, c[0], c[1]) > _cross_product(px, py, qx, qy, left[0],
                                                                                            left[1])):
            left = c
        elif cross < 0.0 and (
                right is None or _cross_product(px, py, qx, qy, c[0], c[1]) < _cross_product(px, py, qx, qy, right[0],
                                                                                             right[1])):
            right = c

    # Select which circle to return
    if left is None and right is None:
        return circ
    elif left is None:
        return right
    elif right is None:
        return left
    else:
        return left if (left[2] <= right[2]) else right


def make_diameter(a, b):
    cx = (a[0] + b[0]) / 2
    cy = (a[1] + b[1]) / 2
    r0 = math.hypot(cx - a[0], cy - a[1])
    r1 = math.hypot(cx - b[0], cy - b[1])
    return (cx, cy, max(r0, r1))


def make_circumcircle(a, b, c):
    # Mathematical algorithm from Wikipedia: Circumscribed circle
    ox = (min(a[0], b[0], c[0]) + max(a[0], b[0], c[0])) / 2
    oy = (min(a[1], b[1], c[1]) + max(a[1], b[1], c[1])) / 2
    ax = a[0] - ox;
    ay = a[1] - oy
    bx = b[0] - ox;
    by = b[1] - oy
    cx = c[0] - ox;
    cy = c[1] - oy
    d = (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) * 2.0
    if d == 0.0:
        return None
    x = ox + ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
    y = oy + ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
    ra = math.hypot(x - a[0], y - a[1])
    rb = math.hypot(x - b[0], y - b[1])
    rc = math.hypot(x - c[0], y - c[1])
    return (x, y, max(ra, rb, rc))


_MULTIPLICATIVE_EPSILON = 1 + 1e-14


def is_in_circle(c, p):
    return c is not None and math.hypot(p[0] - c[0], p[1] - c[1]) <= c[2] * _MULTIPLICATIVE_EPSILON


# Returns twice the signed area of the triangle defined by (x0, y0), (x1, y1), (x2, y2).
def _cross_product(x0, y0, x1, y1, x2, y2):
    return (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)


def numerical_stable_circle(points):
    pts = np.array(points)
    mean_pts = np.mean(pts, 0)
    # print('mean of points:')
    # print(mean_pts)
    pts -= mean_pts  # translate towards origin
    result = make_circle(pts)
    # print('result without mean:')
    # print(result)
    # print('result with mean:')
    # print((result[0] + mean_pts[0], result[1] + mean_pts[1], result[2]))
    x = result[0] + mean_pts[0]
    y = result[1] + mean_pts[1]
    r = result[2]

    return x,y,r

def convert_region_seg_to_circle(regions_seg, newtype):
    for i in range(len(regions_seg)):
        contour_seg = regions_seg[i]
        vertices_seg = contour_seg['Vertices']['Vertex']

        cnt_seg = np.zeros((len(vertices_seg), 2))
        for vi in range(len(vertices_seg)):
            xx = float(vertices_seg[vi]['@X'])
            yy = float(vertices_seg[vi]['@Y'])
            cnt_seg[vi, 0] = xx
            cnt_seg[vi, 1] = yy

        x,y,r = numerical_stable_circle(cnt_seg)
        x = int(x)
        y = int(y)
        r = int(np.ceil(r))

        vertices_circle = []
        vertices_point = OrderedDict()
        vertices_point['@X'] = '%d' % (x - r)
        vertices_point['@Y'] = '%d' % (y - r)
        vertices_circle.append(vertices_point)

        vertices_point2 = OrderedDict()
        vertices_point2['@X'] = '%d' % (x + r)
        vertices_point2['@Y'] = '%d' % (y + r)
        vertices_circle.append(vertices_point2)

        regions_seg[i]['Vertices']['Vertex'] = vertices_circle
        regions_seg[i]['@Type'] = newtype

    return regions_seg


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
    seg_root_dir = '/media/huoy1/48EAE4F7EAE4E264/Projects/fromGe/manual_seg_final'
    circle_root_dir = '/media/huoy1/48EAE4F7EAE4E264/Projects/fromGe/manual_circle_final'

    xml_list = glob.glob(os.path.join(seg_root_dir, '*.xml'))
    xml_list.sort()

    newtype = '2'

    for xi in range(len(xml_list)):
        xml_file = xml_list[xi]
        regions_seg = read_xml(xml_file)
        output_file = xml_file.replace(seg_root_dir,circle_root_dir)

        regions_circle = convert_region_seg_to_circle(regions_seg, newtype)

        write_xml_file(xml_file, regions_circle, output_file)