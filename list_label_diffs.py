"""ラベル補正による差分が多い画像のリストを保存する"""

import os
import cv2
import sys
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
from chainer import cuda
import chainer.functions as F

chainer.config.enable_backprop = False
chainer.config.train = False


def parse_args():
    parser = argparse.ArgumentParser(description='COCO evaluation')
    parser.add_argument('arch', choices=params['archs'].keys(),
                        default='posenet', help='Model architecture')
    parser.add_argument('weights', help='weights file path')
    parser.add_argument('--stages', '-s', type=int, default=6,
                        help='number of stages of posenet')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', default='result/visualization')
    args = parser.parse_args()
    params['inference_img_size'] = params['archs'][args.arch].insize
    params['downscale'] = params['archs'][args.arch].downscale
    params['pad'] = params['archs'][args.arch].pad
    params['inference_scales'] = [1]
    return args


if __name__ == '__main__':
    args = parse_args()

    img_ids_ = []
    heatmap_diffs = []
    paf_diffs = []

    output_dir = args.out
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mode = 'val'  # train, val, eval
    coco = COCO(os.path.join(params['coco_dir'], 'annotations/person_keypoints_{}2017.json'.format(mode)))
    data_loader = CocoDataLoader(params['coco_dir'], coco, params['insize'], mode=mode,
                                 augment_data=False, resize_data=False, use_line_paf=False)

    pose_detector = PoseDetector(args.arch, args.weights, device=args.gpu, precise=True, stages=args.stages)

    for i in range(len(data_loader)):
        """Save ground truth labels"""
        img, img_id, annotations, ignore_mask = data_loader.get_img_annotation(ind=i)

        print(i, len(data_loader), img_id)

        poses = data_loader.parse_coco_annotation(annotations)
        scales = np.ones(len(annotations))

        h, w = img.shape[:2]
        shape = (int(w*params['insize']/h), params['insize']) if h < w else (params['insize'], int(h*params['insize']/w))
        img, ignore_mask, poses = data_loader.resize_data(img, ignore_mask, poses, shape)

        heatmaps = data_loader.gen_heatmaps(img, poses, scales, params['heatmap_sigma'])
        pafs = data_loader.gen_pafs(img, poses, scales, params['paf_sigma'])
        ignore_mask = cv2.morphologyEx(ignore_mask.astype('uint8'), cv2.MORPH_DILATE, np.ones((16, 16))).astype('bool')

        # img, pafs, heatmaps, ignore_mask = data_loader.gen_labels(img, annotations, ignore_mask)

        # resize to view
        shape = img.shape[1::-1]
        pafs = cv2.resize(pafs.transpose(1, 2, 0), shape).transpose(2, 0, 1)
        heatmaps = cv2.resize(heatmaps.transpose(1, 2, 0), shape).transpose(2, 0, 1)
        ignore_mask = cv2.resize(ignore_mask.astype(np.uint8)*255, shape) > 0

        """inference"""
        poses_, scores = pose_detector(img)

        """label completion"""
        xp = cuda.get_array_module(pose_detector.pafs)
        comp_pafs_t = pafs.copy()
        comp_heatmaps_t = heatmaps.copy()
        pafs_teacher = pose_detector.pafs.copy()
        heatmaps_teacher = pose_detector.heatmaps.copy()
        shape = comp_heatmaps_t.shape[1:]
        pafs_teacher = F.resize_images(pafs_teacher[None], shape).data[0]
        heatmaps_teacher = F.resize_images(heatmaps_teacher[None], shape).data[0]
        # pafs
        pafs_t_mag = comp_pafs_t[::2]**2 + comp_pafs_t[1::2]**2
        pafs_t_mag = xp.repeat(pafs_t_mag, 2, axis=0)
        pafs_teacher_mag = pafs_teacher[::2]**2 + pafs_teacher[1::2]**2
        pafs_teacher_mag = xp.repeat(pafs_teacher_mag, 2, axis=0)
        comp_pafs_t[pafs_t_mag < pafs_teacher_mag] = pafs_teacher[pafs_t_mag < pafs_teacher_mag]
        # heatmaps
        comp_heatmaps_t[:, :-1][comp_heatmaps_t[:, :-1] < heatmaps_teacher[:, :-1]] = heatmaps_teacher[:, :-1][comp_heatmaps_t[:, :-1] < heatmaps_teacher[:, :-1]].copy()
        comp_heatmaps_t[:, -1][comp_heatmaps_t[:, -1] > heatmaps_teacher[:, -1]] = heatmaps_teacher[:, -1][comp_heatmaps_t[:, -1] > heatmaps_teacher[:, -1]].copy()

        """append results"""
        img_ids_.append(img_id)
        heatmap_diff = (((comp_heatmaps_t[:-1] - heatmaps[:-1])**2)**.5).mean() / len(annotations)
        heatmap_diffs.append(heatmap_diff)
        paf_diff = (((comp_pafs_t - pafs)**2)**.5).mean() / len(annotations)
        paf_diffs.append(paf_diff)

    """save results"""
    heatmap_img_list = np.array(img_ids_)[np.argsort(heatmap_diffs)][::-1]
    paf_img_list = np.array(img_ids_)[np.argsort(paf_diffs)][::-1]

    txt = '\n'.join(list(map(str, heatmap_img_list)))
    open(os.path.join(output_dir, 'heatmap_img_list.txt'), 'w').write(txt)
    txt = '\n'.join(list(map(str, paf_img_list)))
    open(os.path.join(output_dir, 'paf_img_list.txt'), 'w').write(txt)
