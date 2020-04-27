from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
import eval_protocals.kidpath_circle as kidpath_circle
from eval_protocals.circle_eval import CIRCLEeval
from pycocotools.cocoeval import COCOeval

import numpy as np
import json
import os
import xmltodict

import torch.utils.data as data


class KidPath_FirstBatch_R24(data.Dataset):
    num_classes = 1
    default_resolution = [512, 512]
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, opt):
        super(KidPath_FirstBatch_R24, self).__init__()

        self.max_objs = 128
        self.class_name = [
            '__background__', 'glomerulus']
        self._valid_ids = [1]
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                          for v in range(1, self.num_classes + 1)]
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        # self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
        # self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

        self.circle = kidpath_circle.CIRCLE(opt)

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def xml_to_box(self, xml_file, type='auto', max_det = -1, incrment = 0):

        all_bboxes = []

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

                for j in range(len(regions)):
                    contour = regions[j]
                    vertices = contour['Vertices']['Vertex']

                    center_coordinates_x = ((float(vertices[0]['@X']) + float(vertices[1]['@X'])) / 2.0) + incrment
                    center_coordinates_y = ((float(vertices[0]['@Y']) + float(vertices[1]['@Y'])) / 2.0) + incrment
                    center_coordinates = (center_coordinates_x, center_coordinates_y)
                    radius = np.abs(float(vertices[0]['@X']) - float(vertices[1]['@X'])) / 2.0

                    if type == 'manual':
                        score = 1.0
                    else:
                        score = float(contour['@Text'])
                    bbox = []
                    bbox = [center_coordinates_x, center_coordinates_y, radius, score, 0.0]

                    if max_det == -1 or len(all_bboxes)< max_det:
                        all_bboxes.append(bbox)

        return all_bboxes

    def convert_eval_format(self, all_bboxes):
        # import pdb; pdb.set_trace()
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = self._valid_ids[cls_ind - 1]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = bbox[4]
                    bbox_out = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "score": float("{:.2f}".format(score))
                    }
                    if len(bbox) > 5:
                        extreme_points = list(map(self._to_float, bbox[5:13]))
                        detection["extreme_points"] = extreme_points
                    detections.append(detection)
        return detections

    def convert_eval_circle_format(self, all_circles):
        # import pdb; pdb.set_trace()
        detections = []
        for image_id in all_circles:
            for cls_ind in all_circles[image_id]:
                try:
                    category_id = self._valid_ids[cls_ind - 1]
                except:
                    aaa  =1
                for circle in all_circles[image_id][cls_ind]:
                    score = circle[3]
                    circle_out = list(map(self._to_float, circle[0:3]))

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "score": float("{:.2f}".format(score)),
                        'circle_center': [circle_out[0], circle_out[1]],
                        'circle_radius': circle_out[2]
                    }
                    if len(circle) > 5:
                        extreme_points = list(map(self._to_float, circle[5:13]))
                        detection["extreme_points"] = extreme_points

                    # output_h = 512  # hard coded
                    # output_w = 512  # hard coded
                    # cp = [0, 0]
                    # cp[0] = circle_out[0]
                    # cp[1] = circle_out[1]
                    # cr = circle_out[2]
                    # if cp[0] - cr < 0 or cp[0] + cr > output_w:
                    #     continue
                    # if cp[1] - cr < 0 or cp[1] + cr > output_h:
                    #     continue

                    detections.append(detection)
        return detections

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir):
        json.dump(self.convert_eval_format(results),
                  open('{}/results.json'.format(save_dir), 'w'))

    def run_eval(self, results, save_dir):
        # result_json = os.path.join(save_dir, "results.json")
        # detections  = self.convert_eval_format(results)
        # json.dump(detections, open(result_json, "w"))
        # coco_dets = results
        self.save_results(results, save_dir)
        coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()


    def save_circle_results(self, results, save_dir):  #results save n*m detection results, n is number of images, m is detection number (100) for each image
        json.dump(self.convert_eval_circle_format(results),
                  open('{}/results.json'.format(save_dir), 'w'))

    def run_circle_eval(self, results, save_dir):
        # result is a mxnxdetction dictionary, m is images, n is labels, detection results
        # result_json = os.path.join(save_dir, "results.json")
        # detections  = self.convert_eval_format(results)
        # json.dump(detections, open(result_json, "w"))
        self.save_circle_results(results, save_dir)
        circle_dets = self.circle.loadRes('{}/results.json'.format(save_dir))
        circle_eval = CIRCLEeval(self.circle, circle_dets, "circle")
        # circle_eval = CIRCLEeval(self.circle, circle_dets, "circle_box")
        circle_eval.evaluate()
        circle_eval.accumulate()
        circle_eval.summarize()