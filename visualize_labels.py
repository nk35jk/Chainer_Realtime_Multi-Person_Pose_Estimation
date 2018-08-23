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


"""
GTから生成したラベル, モデルの出力, 補正後のラベル, 推定結果を保存
それらを可視化したHTMLファイルを保存
"""


def parse_args():
    parser = argparse.ArgumentParser(description='COCO evaluation')
    parser.add_argument('arch', choices=params['archs'].keys(),
                        default='posenet', help='Model architecture')
    parser.add_argument('weights', help='weights file path')
    parser.add_argument('--stages', '-s', type=int, default=6,
                        help='number of stages of posenet')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result/paf')
    args = parser.parse_args()
    params['inference_img_size'] = params['archs'][args.arch].insize
    params['downscale'] = params['archs'][args.arch].downscale
    params['pad'] = params['archs'][args.arch].pad
    params['inference_scales'] = [1]
    return args


if __name__ == '__main__':
    args = parse_args()

    # trainのGTのラベルが適切でないもの
    # img_ids = [326, 395, 459]
    # valのGTのラベルが適切でないもの
    # img_ids = [1296, 4395, 11051, 16598, 18193, 48564, 50811, 58705, 60507,
    #            62808, 66771, 70739, 84031, 84674, 93437, 131444, 143572]

    path = 'result/visualization/paf_img_list.txt'
    img_ids = list(map(int, (open(path).read().split())))

    output_dir = args.out
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mode = 'val'  # train, val, eval
    coco = COCO(os.path.join(
        params['coco_dir'], 'annotations/person_keypoints_{}2017.json'.format(mode)))
    data_loader = CocoDataLoader(params['coco_dir'], coco, params['insize'],
                                 mode=mode, augment_data=False,
                                 resize_data=False, use_line_paf=False)

    pose_detector = PoseDetector(args.arch, args.weights, device=args.gpu,
                                 precise=True, stages=args.stages)

    # save css file
    with open(os.path.join(output_dir, 'style.css'), 'w') as f:
        f.write(
         '''body {
              white-space:nowrap;
              font-family: "Hiragino Kaku Gothic ProN"
            }
            span {
              display: inline-block;
              width: 300px;
              text-align: center;
            }
            img {
              width: 300px;
            }
            div {
              text-align: center;
            }'''
        )

    # open html file
    f = open(os.path.join(output_dir, 'visualization.html'), 'w')
    f.write('<head><link rel="stylesheet" type="text/css" href="style.css"></head>')

    for img_id in img_ids[1000:]:
    # for i in range(len(data_loader)):

        """Save ground truth labels"""
        img, img_id, annotations, ignore_mask = data_loader.get_img_annotation(img_id=img_id)
        # img, img_id, annotations, ignore_mask = data_loader.get_img_annotation(ind=i)

        print(img_id)

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

        # overlay labels
        cv2.imwrite(os.path.join(output_dir, '{:08d}.jpg'.format(img_id)), img)
        pafs_to_show = data_loader.overlay_pafs(img.copy(), pafs, .25, .75)
        pafs_to_show = data_loader.overlay_ignore_mask(pafs_to_show, ignore_mask, .5, .5)
        cv2.imwrite(os.path.join(output_dir, '{:08d}_pafs_gt.jpg'.format(img_id)), pafs_to_show)
        heatmaps_to_show = data_loader.overlay_heatmaps(img.copy(), heatmaps[:len(JointType)], .25, .75)
        heatmaps_to_show = data_loader.overlay_ignore_mask(heatmaps_to_show, ignore_mask, .5, .5)
        cv2.imwrite(os.path.join(output_dir, '{:08d}_heatmaps_gt.jpg'.format(img_id)), heatmaps_to_show)
        labels_to_show = data_loader.overlay_pafs(img.copy(), pafs, .4, .6)
        labels_to_show = data_loader.overlay_heatmap(labels_to_show, heatmaps[:len(JointType)].max(axis=0), .6, .4)
        labels_to_show = data_loader.overlay_ignore_mask(labels_to_show, ignore_mask, .5, .5)
        cv2.imwrite(os.path.join(output_dir, '{:08d}_labels_gt.jpg'.format(img_id)), labels_to_show)

        """GT joints"""
        gt_img = img.copy()
        for pose in poses.round().astype('i'):
            for x, y, v in pose:
                if v != 0:
                    cv2.circle(gt_img, (x, y), 6, (0, 255, 0), -1)
        cv2.imwrite(os.path.join(output_dir, '{:08d}_gt_joints.jpg'.format(img_id)), gt_img)

        """inference"""
        poses, scores = pose_detector(img)

        """evaluation"""
        res = []
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

        if len(res) == 0:
            ap = -1
        else:
            cocoDt = coco.loadRes(res)
            cocoEval = COCOeval(coco, cocoDt, 'keypoints')
            cocoEval.params.imgIds = [img_id]
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            ap = cocoEval.stats[0]

        # """Save output heatmaps and pafs (channel-wise)"""
        # # heatmaps
        # for i, heatmap in enumerate(pose_detector.heatmaps):
        #     rgb_heatmap = cv2.applyColorMap((heatmap.clip(0, 1)*255).astype(np.uint8), cv2.COLORMAP_JET)
        #     heatmap_img = cv2.addWeighted(img, 0.3, rgb_heatmap, 0.7, 0)
        #     cv2.imwrite(os.path.join(output_dir, '{:08d}_heatmap{}.jpg'.format(img_id, i)), heatmap_img)
        #
        # # pafs (absolute value)
        # abs_pafs = (pose_detector.pafs[::2]**2 + pose_detector.pafs[1::2]**2)**0.5
        # for i, abs_paf in enumerate(abs_pafs):
        #     rgb_paf = cv2.applyColorMap((abs_paf.clip(0, 1)*255).astype(np.uint8), cv2.COLORMAP_JET)
        #     paf_img = cv2.addWeighted(img, 0.3, rgb_paf, 0.7, 0)
        #     cv2.imwrite(os.path.join(output_dir, '{:08d}_abs_paf{}.jpg'.format(img_id, i)), paf_img)

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

        # overlay labels
        pafs_to_show = data_loader.overlay_pafs(img.copy(), pafs_teacher)
        # heatmaps_to_show = data_loader.overlay_heatmap(img.copy(), heatmaps_teacher[:-1].max(axis=0))
        heatmaps_to_show = data_loader.overlay_heatmaps(img.copy(), heatmaps_teacher[:len(JointType)], .25, .75)
        cv2.imwrite(os.path.join(output_dir, '{:08d}_pafs.jpg'.format(img_id)), pafs_to_show)
        cv2.imwrite(os.path.join(output_dir, '{:08d}_heatmaps.jpg'.format(img_id)), heatmaps_to_show)

        comp_pafs_to_show = data_loader.overlay_pafs(img.copy(), comp_pafs_t)
        cv2.imwrite(os.path.join(output_dir, '{:08d}_pafs_comp.jpg'.format(img_id)), comp_pafs_to_show)
        # comp_heatmpas_to_show = data_loader.overlay_heatmap(img.copy(), comp_heatmaps_t[:-1].max(axis=0))
        comp_heatmpas_to_show = data_loader.overlay_heatmaps(img.copy(), comp_heatmaps_t[:len(JointType)], .25, .75)
        cv2.imwrite(os.path.join(output_dir, '{:08d}_heatmaps_comp.jpg'.format(img_id)), comp_heatmpas_to_show)

        # cv2.imshow('w', np.hstack([img, comp_pafs_to_show]))
        # k = cv2.waitKey(1)
        # if k == ord('q'):
        #     sys.exit()
        # elif k == ord('d'):
        #     import ipdb; ipdb.set_trace()

        """heatmap peaks"""
        peaks_img = img.copy()
        for all_peak in pose_detector.all_peaks:
            cv2.circle(peaks_img, (int(all_peak[1]), int(all_peak[2])), 4, params['joint_colors'][int(all_peak[0])], -1)
        cv2.imwrite(os.path.join(output_dir, '{:08d}_peaks.jpg'.format(img_id)), peaks_img)

        """final results"""
        result_img = draw_person_pose(img, poses)
        cv2.imwrite(os.path.join(output_dir, '{:08d}_final_results.jpg'.format(img_id)), result_img)

        """write html"""
        f.write('<div>')
        # for i in range(len(pose_detector.heatmaps)):
        #     f.write('<img src="{}">'.format('{:08d}_heatmap{}.jpg'.format(img_id, i)))
        # for i in range(len(pose_detector.pafs)):
        #     f.write('<img src="{}">'.format('{:08d}_abs_paf{}.jpg'.format(img_id, i)))
        f.write(
            # '<img src="{}">'.format('{:08d}.jpg'.format(img_id))
            '<img src="{}">'.format('{:08d}_gt_joints.jpg'.format(img_id))
            + '<img src="{}">'.format('{:08d}_labels_gt.jpg'.format(img_id))
            + '<img src="{}">'.format('{:08d}_heatmaps_gt.jpg'.format(img_id))
            + '<img src="{}">'.format('{:08d}_heatmaps.jpg'.format(img_id))
            + '<img src="{}">'.format('{:08d}_heatmaps_comp.jpg'.format(img_id))
            + '<img src="{}">'.format('{:08d}_pafs_gt.jpg'.format(img_id))
            + '<img src="{}">'.format('{:08d}_pafs.jpg'.format(img_id))
            + '<img src="{}">'.format('{:08d}_pafs_comp.jpg'.format(img_id))
            # + '<img src="{}">'.format('{:08d}_peaks.jpg'.format(img_id))
            # + '<img src="{}">'.format('{:08d}_final_results.jpg'.format(img_id))
            + '\n'
        )
        f.write('</div>')
        f.write('<div>')
        # for jt in JointType:
        #     f.write('<span>{}</span>'.format(str(jt).split('.')[-1]))
        # f.write('<span>All joints</span>')
        # for limb in params['limbs_point']:
        #     txt = '<span>{}-{}</span>'.format(str(limb[0]).split('.')[-1], str(limb[1]).split('.')[-1])
        #     f.write(txt * 2)

        # f.write('<span>image</span>')
        f.write('<span>GT</span>')
        f.write('<span>GT labels</span>')
        f.write('<span>GT heatmaps</span>')
        f.write('<span>Output heatmaps</span>')
        f.write('<span>Completed heatmaps</span>')
        f.write('<span>GT PAFs</span>')
        f.write('<span>Output PAFs</span>')
        f.write('<span>Completed PAFs</span>')
        # f.write('<span>peaks</span>')
        # f.write('<span>Final results, AP: {:.3f}</span>'.format(ap))
        f.write('</div>')

    f.close()
