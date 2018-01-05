import os
import cv2
import time
import json
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

from hyperopt import hp, tpe, Trials, fmin

chainer.config.enable_backprop = False
chainer.config.train = False


def parse_args():
    parser = argparse.ArgumentParser(description='COCO evaluation')
    parser.add_argument('arch', choices=params['archs'].keys(), default='posenet', help='Model architecture')
    parser.add_argument('weights', help='weights file path')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--precise', action='store_true', default=True, help='visualize results')
    parser.add_argument('--mask', action='store_true')
    parser.add_argument('--out', '-o', default='result', help='Output directory')
    args = parser.parse_args()
    params['inference_img_size'] = params['archs'][args.arch].insize
    params['downscale'] = params['archs'][args.arch].downscale
    params['pad'] = params['archs'][args.arch].pad
    return args


class Objective(object):

    def __init__(self):
        self.coco_val = COCO(os.path.join(params['coco_dir'], 'annotations/person_keypoints_val2017.json'))
        self.eval_loader = CocoDataLoader(self.coco_val, params['inference_img_size'], mode='eval', n_samples=None)
        self.pose_detector = PoseDetector(args.arch, args.weights, device=args.gpu, precise=args.precise, compute_mask=args.mask)

    def __call__(self, p):
        for key in p.keys():
            params[key] = p[key]

        res = []
        imgIds = []
        for i in range(2000, 2100):
            img, annotations, img_id = self.eval_loader.get_example(i)

            imgIds.append(img_id)

            st = time.time()
            poses, scores = self.pose_detector(img)
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

        try:
            cocoDt = self.coco_val.loadRes(res)
            cocoEval = COCOeval(self.coco_val, cocoDt, 'keypoints')
            cocoEval.params.imgIds = imgIds
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            ap = cocoEval.stats[0]

            path = os.path.join(args.out, 'optimize_log_{}.csv'.format(args.arch))
            with open(path, 'a') as f:
                f.write('{}, {}, {}, {}, {}, {}, {}\n'.format(
                    ap,
                    p['n_integ_points'],
                    p['n_integ_points_thresh'],
                    p['heatmap_peak_thresh'],
                    p['inner_product_thresh'],
                    p['limb_length_ratio'],
                    p['subset_score_thresh'],
                ))
        except:
            ap = 0
        return -ap


if __name__ == '__main__':
    args = parse_args()

    objective = Objective()

    space = {
        'n_integ_points': hp.quniform('n_integ_points', 10, 15, 1),  # 10
        'n_integ_points_thresh': hp.quniform('n_integ_points_thresh', 3, 8, 1),  # 8
        'heatmap_peak_thresh': hp.uniform('heatmap_peak_thresh', 0.005, 0.3),  # 0.1
        'inner_product_thresh': hp.uniform('inner_product_thresh', 0.001, 0.1),  # 0.05
        'limb_length_ratio': hp.uniform('limb_length_ratio', 0.1, 1.5),  # 0.5
        'n_subset_limbs_thresh': hp.quniform('n_subset_limbs_thresh', 3, 10, 1),  # 7
        'subset_score_thresh': hp.uniform('subset_score_thresh', 0.05, 1),  # 0.4
    }

    max_evals = 1000

    trials = Trials()

    best = fmin(
        objective,
        space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        verbose=1
    )

    best_ap = -trials.best_trial['result']['loss']

    path = os.path.join(args.out, 'params_{}_ap_{:.1f}.json'.format(args.arch, best_ap*100))
    with open(path, 'w') as f:
        json.dump(best, f)
