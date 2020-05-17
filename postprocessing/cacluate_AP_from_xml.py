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
import matplotlib.pyplot as plt

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

def save_det_as_txt(det, outdir, ftype = 'detection',maxDets = 99999):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for fi in range(len(det)):
        fname = '%05d.txt' % (fi+1)
        fpath = os.path.join(outdir, fname)

        outF = open(fpath, "w")

        for li in range(len(det[fi][1])):
            if li>=maxDets:
                continue
            # write line to output file
            if ftype == 'detection':
                line = 'glumerulus %0.4f %.4f %.4f %.4f' % (det[fi][1][li][3], det[fi][1][li][0], det[fi][1][li][1], det[fi][1][li][2])
            elif ftype == 'groundtruth':
                line = 'glumerulus %.4f %.4f %.4f' % (det[fi][1][li][0], det[fi][1][li][1], det[fi][1][li][2])
            outF.write(line)
            outF.write("\n")
        outF.close()


if __name__ == "__main__":

    sublist = ['25119', '24739', '24738', '23681', '23499']

    # sublist = ['25119', '24739']

    # manual_xml_dir = '/media/huoy1/48EAE4F7EAE4E264/Projects/from_haichun/R24_scan_slides_manual_QA'
    manual_xml_dir = '/media/huoy1/48EAE4F7EAE4E264/Projects/fromGe/manual_circle_final'
    # #use QA data
    auto_xml_dirs = []
    evaluate_dirs = []
    exp_strs = []

    auto_xml_dir = '/media/huoy1/48EAE4F7EAE4E264/Projects/from_haichun/detection_results/kidney_first_batch_R24_dla_34'
    evaluate_dir = '/media/huoy1/48EAE4F7EAE4E264/Projects/from_haichun/detection_results/Pipeline_paper/Results/kidney_first_batch_R24_dla_34'
    exp_str = 'CircleNet (with new annotations)'
    auto_xml_dirs.append(auto_xml_dir)
    evaluate_dirs.append(evaluate_dir)
    exp_strs.append(exp_str)


    # #use raw data
    auto_xml_dir = '/media/huoy1/48EAE4F7EAE4E264/Projects/from_haichun/detection_results/kidney_first_batch_dla_34'
    evaluate_dir = '/media/huoy1/48EAE4F7EAE4E264/Projects/from_haichun/detection_results/Pipeline_paper/Results/kidney_first_batch_dla_34'
    exp_str = 'CircleNet'
    auto_xml_dirs.append(auto_xml_dir)
    evaluate_dirs.append(evaluate_dir)
    exp_strs.append(exp_str)

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

    recalls = []
    precisions = []

    for ei in range(len(auto_xml_dirs)):
        auto_xml_dir = auto_xml_dirs[ei]
        evaluate_dir = evaluate_dirs[ei]

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

        auto_dir = os.path.join(evaluate_dir, 'detection')
        save_det_as_txt(auto_bboxs, auto_dir,'detection', 1000)

        manual_dir = os.path.join(evaluate_dir, 'groundtruth')
        save_det_as_txt(manual_bboxs, manual_dir,'groundtruth')

        imgIds = np.array(range(len(sublist))) + 1

        circle_eval = CIRCLEeval(manual_det, auto_det, "circle", imgIds)
        # circle_eval = CIRCLEeval(self.circle, circle_dets, "circle_box")
        circle_eval.evaluate()
        circle_eval.accumulate()
        circle_eval.summarize()

        precision = circle_eval.eval['precision'][0, :, 0, 0, 2]
        recall = circle_eval.params.recThrs

        recalls.append(recall)
        precisions.append(precision)


    plt.plot(recalls[0], precisions[0], label='Precision')
    plt.plot(recalls[1], precisions[1], label='Precision')
    plt.xlabel('recall')
    plt.ylabel('precision')


    # ap_str = "{0:.3f}".format(np.mean(precision) )
    # ap_str = "{0:.4f}%".format(average_precision * 100)
    plt.title('Precision x Recall curve')

    plt.legend(exp_strs, shadow=True)
    plt.grid()
    plt.show()

    def CalculateAveragePrecision(rec, prec):
        mrec = []
        mrec.append(0)
        [mrec.append(e) for e in rec]
        mrec.append(1)
        mpre = []
        mpre.append(0)
        [mpre.append(e) for e in prec]
        mpre.append(0)
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        ii = []
        for i in range(len(mrec) - 1):
            if mrec[1:][i] != mrec[0:-1][i]:
                ii.append(i + 1)
        ap = 0
        for i in ii:
            ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
        # return [ap, mpre[1:len(mpre)-1], mrec[1:len(mpre)-1], ii]
        return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]


    [ap, mpre, mrec, ii] = CalculateAveragePrecision(recall[0],precision[0])
    print(ap)
