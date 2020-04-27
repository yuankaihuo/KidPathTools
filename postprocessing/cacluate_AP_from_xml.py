import openslide
import xmltodict
import numpy as np
from PIL import Image
import os
import cv2
from scn_to_png import scn_to_png, scn_to_png_whole_slide, save_cropped_img_mask, scn_to_png_atubular, scn_to_png_whole_slide_good_bad, save_cropped_img_mask_good_bad, save_cropped_img_mask_bad
import glob
from kidpath import KidPath_FirstBatch_R24
from eval_protocals.circle_eval import CIRCLEeval

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

if __name__ == "__main__":

    sublist = ['25119', '24739', '24738', '23681', '23499']

    manual_xml_dir = '/media/huoy1/48EAE4F7EAE4E264/Projects/from_haichun/R24_scan_slides_manual_QA'
    # #use QA data
    # auto_xml_dir = '/media/huoy1/48EAE4F7EAE4E264/Projects/from_haichun/detection_results/kidney_first_batch_R24_dla_34'
    # #use raw data
    auto_xml_dir = '/media/huoy1/48EAE4F7EAE4E264/Projects/from_haichun/detection_results/kidney_first_batch_dla_34'

    CATEGORIES = [
        {
            'id': 1,
            'name': 'glomerulus',
            'supercategory': 'glomerulus',
        }
    ]

    opt = {}
    opt['categories'] = CATEGORIES
    det_obj = KidPath_FirstBatch_R24(opt)


    auto_bboxs = {}
    manual_bboxs = {}
    for fi in range(len(sublist)):
        subname = sublist[fi]
        # get all manual detection ground truth
        manual_xml_files = glob.glob(os.path.join(manual_xml_dir, '%s*.xml'%subname))
        assert len(manual_xml_files) == 1
        manual_xml_file = manual_xml_files[0]

        manual_xml_name = os.path.basename(manual_xml_file).replace('.xml','')
        auto_xml_file = os.path.join(auto_xml_dir, manual_xml_name, manual_xml_name+'.xml')
        if not os.path.exists(auto_xml_file):
            auto_xml_file = os.path.join(auto_xml_dir, manual_xml_name + '.xml')

        assert os.path.exists(auto_xml_file)

        auto_bboxs[fi] = {}
        auto_bboxs[fi][1] = det_obj.xml_to_box(auto_xml_file, 'auto')
        manual_bboxs[fi] = {}
        manual_bboxs[fi][1] = det_obj.xml_to_box(manual_xml_file, 'manual')


    auto_list = det_obj.convert_eval_circle_format(auto_bboxs)
    manual_list = det_obj.convert_eval_circle_format(manual_bboxs)

    auto_det = det_obj.circle.loadRes(auto_list)
    manual_det = det_obj.circle.loadRes(manual_list)

    imgIds = np.array(range(len(sublist))) + 1

    circle_eval = CIRCLEeval(manual_det, auto_det, "circle", imgIds)
    # circle_eval = CIRCLEeval(self.circle, circle_dets, "circle_box")
    circle_eval.evaluate()
    circle_eval.accumulate()
    circle_eval.summarize()

