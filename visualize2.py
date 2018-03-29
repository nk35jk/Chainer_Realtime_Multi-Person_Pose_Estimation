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


def overlay_heatmap(img, paf):
    hue = (np.arctan2(paf[1], paf[0]) / np.pi) / -2 + 0.5
    saturation = np.ones_like(hue)
    value = np.sqrt(paf[0] ** 2 + paf[1] ** 2) * 2  # twiced
    value[value > 1.0] = 1.0
    hsv_paf = np.stack([hue*180, saturation*255, value*255]).transpose(1, 2, 0)
    rgb_paf = cv2.cvtColor(hsv_paf.astype(np.uint8), cv2.COLOR_HSV2BGR)
    img = cv2.addWeighted(img, 0.25, rgb_paf, 0.75, 0)
    return img

def overlay_heatmaps(img, heatmaps):
    mix_heatmap = np.zeros((2,) + img.shape[:-1])
    paf_flags = np.zeros(mix_heatmap.shape) # for constant paf

    for paf in heatmaps.reshape((int(pafs.shape[0]/2), 2,) + heatmaps.shape[1:]):
        paf_flags = paf != 0
        paf_flags += np.broadcast_to(paf_flags[0] | paf_flags[1], paf.shape)
        mix_heatmap += paf

    mix_heatmap[paf_flags > 0] /= paf_flags[paf_flags > 0]
    img = overlay_heatmap(img, mix_heatmap)
    return img


def parse_args():
    parser = argparse.ArgumentParser(description='COCO evaluation')
    parser.add_argument('arch', choices=params['archs'].keys(), default='posenet', help='Model architecture')
    parser.add_argument('weights', help='weights file path')
    parser.add_argument('--stages', '-s', type=int, default=6, help='number of posenet stages')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--mask', action='store_true')
    parser.add_argument('--precise', action='store_true', default=True, help='precise inference')
    args = parser.parse_args()
    params['inference_img_size'] = params['archs'][args.arch].insize
    params['downscale'] = params['archs'][args.arch].downscale
    params['pad'] = params['archs'][args.arch].pad
    params['inference_scales'] = [1]
    return args


if __name__ == '__main__':
    args = parse_args()

    params['min_keypoints'] = 5
    params['min_area'] = 32 * 32
    params['insize'] = 368
    params['paf_sigma'] = 8
    params['heatmap_sigma'] = 7
    params['min_scale'] = 1
    params['max_scale'] = 1
    params['max_rotate_degree'] = 0
    params['center_perterb_max'] = 0

    # img_ids = [326, 395, 459]  # trainのGTのラベルが適切でなさそうなもの
    img_ids = [1296, 4395, 11051, 16598, 18193, 48564, 50811, 58705, 60507,
               62808, 66771, 70739, 84031, 84674, 93437, 131444, 143572]  # valのGTのラベルが適切でなさそうなもの

    output_dir = 'result/model_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    label_output_dir = 'result/label'
    if not os.path.exists(label_output_dir):
        os.makedirs(label_output_dir)

    mode = 'val'  # train, val, eval
    coco = COCO(os.path.join(params['coco_dir'], 'annotations/person_keypoints_{}2017.json'.format(mode)))
    data_loader = CocoDataLoader(params['coco_dir'], coco, params['insize'], mode=mode)

    pose_detector = PoseDetector(args.arch, args.weights, device=args.gpu, precise=args.precise, stages=args.stages)

    cv2.namedWindow('w', cv2.WINDOW_NORMAL)

    for img_id in img_ids:
        """GT"""
        img, img_id, annotations, ignore_mask = data_loader.get_img_annotation(img_id=img_id)

        # print('\r{}'.format(img_id), end='')

        poses = data_loader.parse_coco_annotation(annotations)
        h, w = img.shape[:2]
        if h > w:
            out_h = h * params['insize'] // w
            out_w = params['insize']
        else:
            out_w = w * params['insize'] // h
            out_h = params['insize']
        resized_img, ignore_mask, resized_poses = data_loader.resize_data(img, ignore_mask, poses, shape=(out_w, out_h))

        heatmaps = data_loader.generate_heatmaps(resized_img, resized_poses, params['heatmap_sigma'])
        pafs = data_loader.generate_pafs(resized_img, resized_poses, params['paf_sigma'])
        ignore_mask = cv2.morphologyEx(ignore_mask.astype('uint8'), cv2.MORPH_DILATE, np.ones((16, 16))).astype('bool')

        # resize to view
        shape = tuple(reversed(resized_img.shape[:2]))
        pafs = cv2.resize(pafs.transpose(1, 2, 0), shape).transpose(2, 0, 1)
        heatmaps = cv2.resize(heatmaps.transpose(1, 2, 0), shape).transpose(2, 0, 1)
        ignore_mask = cv2.resize(ignore_mask.astype(np.uint8)*255, shape) > 0

        # overlay labels
        img_to_show = resized_img.copy()
        # img_to_show = data_loader.overlay_ignore_mask(img_to_show, ignore_mask)
        img_to_show = data_loader.overlay_pafs(img_to_show, pafs)
        cv2.imwrite(os.path.join(label_output_dir, '{:08d}_gt_paf.jpg'.format(img_id)), img_to_show)
        img_to_show = data_loader.overlay_heatmap(img_to_show, heatmaps[:-1].max(axis=0))

        cv2.imshow('w', np.hstack([resized_img, img_to_show]))
        cv2.imwrite(os.path.join(label_output_dir, '{:08d}_gt.jpg'.format(img_id)), img_to_show)
        cv2.imwrite(os.path.join(label_output_dir, '{:08d}.jpg'.format(img_id)), resized_img)
        # k = cv2.waitKey(1)
        # if k == ord('q'):
        #     sys.exit()
        # elif k == ord('d'):
        #     import ipdb; ipdb.set_trace()

        poses, scores = pose_detector(img)

        """Save results and GT"""
        # heatmaps
        for i, heatmap in enumerate(pose_detector.heatmaps):
            rgb_heatmap = cv2.applyColorMap((heatmap.clip(0, 1)*255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_img = cv2.addWeighted(img, 0.3, rgb_heatmap, 0.7, 0)
            cv2.imwrite(os.path.join(output_dir, '{:08d}_heatmap{}.jpg'.format(img_id, i)), heatmap_img)

        # # pafs (each)
        # for i, paf in enumerate(pose_detector.pafs):
        #     rgb_paf = cv2.applyColorMap((paf.clip(-1, 1)*127.5+127.5).astype(np.uint8), cv2.COLORMAP_JET)
        #     paf_img = cv2.addWeighted(img, 0.3, rgb_paf, 0.7, 0)
        #     cv2.imwrite(os.path.join(output_dir, '{:08d}_paf{}.jpg'.format(img_id, i)), paf_img)

        # pafs (absolute value)
        abs_pafs = (pose_detector.pafs[::2]**2 + pose_detector.pafs[1::2]**2)**0.5
        for i, abs_paf in enumerate(abs_pafs):
            rgb_paf = cv2.applyColorMap((abs_paf.clip(0, 1)*255).astype(np.uint8), cv2.COLORMAP_JET)
            paf_img = cv2.addWeighted(img, 0.3, rgb_paf, 0.7, 0)
            cv2.imwrite(os.path.join(output_dir, '{:08d}_abs_paf{}.jpg'.format(img_id, i)), paf_img)

        """ラベル補正"""
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

        # overlay labels
        comp_img_to_show = resized_img.copy()
        # img_to_show = data_loader.overlay_ignore_mask(comp_img_to_show, ignore_mask)
        comp_img_to_show = data_loader.overlay_pafs(comp_img_to_show, comp_pafs_t)  # pafs_teacher comp_pafs_t
        cv2.imwrite(os.path.join(label_output_dir, '{:08d}_comp_paf.jpg'.format(img_id)), comp_img_to_show)
        comp_img_to_show = data_loader.overlay_heatmap(comp_img_to_show, comp_heatmaps_t[:-1].max(axis=0))

        cv2.imshow('w', np.hstack([resized_img, comp_img_to_show]))
        cv2.imwrite(os.path.join(label_output_dir, '{:08d}_comp.jpg'.format(img_id)), comp_img_to_show)
        k = cv2.waitKey(1)
        if k == ord('q'):
            sys.exit()
        elif k == ord('d'):
            import ipdb; ipdb.set_trace()

        """heatmap peaks"""
        # joint_colors = [
        #     [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
        #     [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
        #     [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
        #     [255, 0, 255], [255, 0, 170], [255, 0, 85]]
        # peaks_img = img.copy()
        # for all_peak in pose_detector.all_peaks:
        #     cv2.circle(peaks_img, (int(all_peak[1]), int(all_peak[2])), 2, joint_colors[int(all_peak[0])], -1)
        # cv2.imwrite(os.path.join(output_dir, '{:08d}_peaks.jpg'.format(img_id)), peaks_img)

        """final results"""
        # result_img = draw_person_pose(img, poses)
        # cv2.imwrite(os.path.join(output_dir, '{:08d}_final_results.jpg'.format(img_id)), result_img)

        """GT joints"""
        # gt_img = img.copy()
        # for ann in annotations:
        #     for x, y, v in np.array(ann['keypoints']).reshape(-1, 3):
        #         if v == 1:
        #             cv2.circle(gt_img, (x, y), 3, (255, 255, 0), -1)
        #         elif v == 2:
        #             cv2.circle(gt_img, (x, y), 3, (255, 0, 255), -1)
        # cv2.imwrite(os.path.join(output_dir, '{:08d}_gt.jpg'.format(img_id)), gt_img)
