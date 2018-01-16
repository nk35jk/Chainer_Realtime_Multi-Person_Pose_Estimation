import os
import cv2
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from coco_data_loader import CocoDataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from entity import JointType, params
from pose_detector import PoseDetector
from pose_detector import draw_person_pose

import chainer

chainer.config.enable_backprop = False
chainer.config.train = False


def parse_args():
    parser = argparse.ArgumentParser(description='COCO evaluation')
    parser.add_argument('arch', choices=params['archs'].keys(), default='posenet', help='Model architecture')
    parser.add_argument('weights', help='weights file path')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--mask', action='store_true')
    parser.add_argument('--precise', action='store_true', default=True, help='precise inference')
    args = parser.parse_args()
    params['inference_img_size'] = params['archs'][args.arch].insize
    params['downscale'] = params['archs'][args.arch].downscale
    params['pad'] = params['archs'][args.arch].pad
    params['inference_scales'] = [1]
    return args


def test():
    output_dir = '{}/vis'.format('/'.join(args.weights.split('/')[:-1]))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pose_detector = PoseDetector(args.arch, args.weights, device=args.gpu, precise=args.precise, compute_mask=args.mask)

    coco_val = COCO(os.path.join(params['coco_dir'], 'annotations/person_keypoints_val2017.json'))
    eval_loader = CocoDataLoader(coco_val, pose_detector.model.insize, mode='eval', n_samples=None)

    f = open(os.path.join(output_dir, 'vis.html'), 'w')
    f.write('<head><link rel="stylesheet" type="text/css" href="style.css"></head>')

    all_res = []
    imgIds = []
    # for i in range(len(eval_loader)):
    for i in range(100):
        res = []

        img, annotations, img_id = eval_loader.get_example(i)
        print('{:4d}, img id = {}'.format(i, img_id))

        imgIds.append(img_id)

        st = time.time()
        poses, scores = pose_detector(img)
        print('inference: {:.2f}s'.format(time.time() - st))

        for pose, score in zip(poses, scores):
            res_dict = {}
            res_dict['category_id'] = 1
            res_dict['image_id'] = img_id
            res_dict['score'] = score * sum(pose[:, 2] > 0)

            keypoints = np.zeros((len(params['coco_joint_indices']), 3))
            for joint, jt in zip(pose, JointType):
                if joint is not None and jt in params['coco_joint_indices']:
                    j = params['coco_joint_indices'].index(jt)
                    keypoints[j] = joint
            res_dict['keypoints'] = keypoints.ravel()
            res.append(res_dict)

        # evaluate single image
        if len(res) == 0:
            ap = -1
        else:
            cocoDt = coco_val.loadRes(res)
            cocoEval = COCOeval(coco_val, cocoDt, 'keypoints')
            cocoEval.params.imgIds = [img_id]
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            ap = cocoEval.stats[0]

        # import IPython; IPython.embed()

        """Save results and GT"""
        # heatmaps
        for i, heatmap in enumerate(pose_detector.heatmaps):
            rgb_heatmap = cv2.applyColorMap((heatmap.clip(0, 1)*255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_img = cv2.addWeighted(img, 0.3, rgb_heatmap, 0.7, 0)
            cv2.imwrite(os.path.join(output_dir, '{:08d}_heatmap{}.jpg'.format(img_id, i)), heatmap_img)

        # pafs
        for i, paf in enumerate(pose_detector.pafs):
            rgb_paf = cv2.applyColorMap((paf.clip(-1, 1)*127.5+127.5).astype(np.uint8), cv2.COLORMAP_JET)
            paf_img = cv2.addWeighted(img, 0.3, rgb_paf, 0.7, 0)
            cv2.imwrite(os.path.join(output_dir, '{:08d}_paf{}.jpg'.format(img_id, i)), paf_img)

        # heatmap peaks
        joint_colors = [
            [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
            [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
            [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
            [255, 0, 255], [255, 0, 170], [255, 0, 85]]
        peaks_img = img.copy()
        for all_peak in pose_detector.all_peaks:
            cv2.circle(peaks_img, (int(all_peak[1]), int(all_peak[2])), 2, joint_colors[int(all_peak[0])], -1)
        cv2.imwrite(os.path.join(output_dir, '{:08d}_peaks.jpg'.format(img_id)), peaks_img)

        # final results
        result_img = draw_person_pose(img, poses)
        cv2.imwrite(os.path.join(output_dir, '{:08d}_final_results.jpg'.format(img_id)), result_img)

        # GT joints
        gt_img = img.copy()
        for ann in annotations:
            for x, y, v in np.array(ann['keypoints']).reshape(-1, 3):
                if v == 1:
                    cv2.circle(gt_img, (x, y), 3, (255, 255, 0), -1)
                elif v == 2:
                    cv2.circle(gt_img, (x, y), 3, (255, 0, 255), -1)
        cv2.imwrite(os.path.join(output_dir, '{:08d}_gt.jpg'.format(img_id)), gt_img)

        # write html
        f.write('<div>')
        for jt in JointType:
            f.write('<span>{}</span>'.format(str(jt).split('.')[-1]))
        f.write('<span>All joints</span>')
        for limb in params['limbs_point']:
            txt = '<span>{}-{}</span>'.format(str(limb[0]).split('.')[-1], str(limb[1]).split('.')[-1])
            f.write(txt * 2)
        f.write('<span>peaks</span>')
        f.write('<span>Final results, AP: {:.3f}</span>'.format(ap))
        f.write('<span>GT</span>')
        f.write('</div>')

        f.write('<div>')
        for i in range(len(pose_detector.heatmaps)):
            f.write('<img src="{}">'.format('{:08d}_heatmap{}.jpg'.format(img_id, i)))
        for i in range(len(pose_detector.pafs)):
            f.write('<img src="{}">'.format('{:08d}_paf{}.jpg'.format(img_id, i)))
        f.write(
              '<img src="{}">'.format('{:08d}_peaks.jpg'.format(img_id))
            + '<img src="{}">'.format('{:08d}_final_results.jpg'.format(img_id))
            + '<img src="{}">'.format('{:08d}_gt.jpg'.format(img_id))
            + '\n'
        )
        f.write('</div>')
    f.close()


if __name__ == '__main__':
    args = parse_args()
    test()
