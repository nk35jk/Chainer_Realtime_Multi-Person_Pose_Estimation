import os
import sys
import json
import time
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt

from coco_data_loader import CocoDataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
# from pycocotools.cocoanalyze import COCOanalyze
from cocoanalyze import COCOanalyze

from entity import JointType, params
from pose_detector import PoseDetector
from pose_detector import draw_person_pose

import chainer

chainer.config.enable_backprop = False
chainer.config.train = False


def parse_args():
    parser = argparse.ArgumentParser(description='COCO evaluation')
    parser.add_argument('--arch', '-a', choices=params['archs'].keys(),
                        default='posenet', help='Model architecture')
    parser.add_argument('--weights', '-w', help='weights file path')
    parser.add_argument('--stages', '-s', type=int, default=6,
                        help='number of stages of posenet')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--display', action='store_true',
                        help='visualize inference results')
    parser.add_argument('--res')
    parser.add_argument('--fast', action='store_false')
    args = parser.parse_args()
    params['inference_img_size'] = params['archs'][args.arch].insize
    params['downscale'] = params['archs'][args.arch].downscale
    params['pad'] = params['archs'][args.arch].pad
    return args


def evaluate():
    save_dir = os.path.dirname(args.weights or args.res)

    coco_val = COCO(os.path.join(params['coco_dir'],
                                 'annotations/person_keypoints_val2017.json'))
    if args.res:
        res = json.load(open(args.res))
        imgIds = [x['image_id'] for x in res]
    else:
        pose_detector = PoseDetector(args.arch, args.weights, device=args.gpu,
                                     precise=args.fast, stages=args.stages)
        eval_loader = CocoDataLoader(params['coco_dir'], coco_val,
                                     params['inference_img_size'], mode='eval')

        res = []
        imgIds = []
        for i in range(args.n_samples):
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
                res_dict['keypoints'] = list(keypoints.ravel())
                res.append(res_dict)

            if args.display:
                img = draw_person_pose(img, poses)
                cv2.imshow('results', img)
                k = cv2.waitKey(1)
                if k == ord('q'):
                    sys.exit()

        with open(os.path.join(os.path.dirname(args.weights), 'res.json'), 'w') as f:
            json.dump(res, f, sort_keys=True, indent=4)

    cocoDt = coco_val.loadRes(res)
    cocoEval = COCOeval(coco_val, cocoDt, 'keypoints')
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    ap = cocoEval.stats[0]

    coco_analyze = COCOanalyze(coco_val, cocoDt, 'keypoints')
    coco_analyze.cocoEval.params.imgIds = imgIds

    # coco_analyze.evaluate(verbose=True, makeplots=True)

    # set OKS threshold of the extended error analysis
    # coco_analyze.params.oksThrs = [.5,.55,.6,.65,.7,.75,.8,.85,.9,.95]
    coco_analyze.params.oksThrs = [.5, .75]


    # set OKS threshold required to match a detection to a ground truth
    coco_analyze.params.oksLocThrs = .1

    # set KS threshold limits defining jitter errors
    coco_analyze.params.jitterKsThrs = [.5, .85]

    # set the localization errors to analyze and in what order
    # note: different order will show different progressive improvement
    # to study impact of single error type, study in isolation
    coco_analyze.params.err_types = ['miss', 'swap' , 'inversion', 'jitter']
    # coco_analyze.params.err_types = ['jitter']

    # area ranges for evaluation
    # 'all' range is union of medium and large
    coco_analyze.params.areaRng       = [[32 ** 2, 1e5 ** 2]] #[96 ** 2, 1e5 ** 2],[32 ** 2, 96 ** 2]
    coco_analyze.params.areaRngLbl    = ['all'] # 'large','medium'

    coco_analyze.params.maxDets = [20]

    coco_analyze.analyze(check_kpts=True, check_scores=False, check_bckgd=True)
    coco_analyze.summarize(makeplots=True, savedir=save_dir)

    path = os.path.join(save_dir, 'evaluation_results.txt')
    if not args.res:
        with open(path, 'a') as f:
            f.write('{}, {}\n'.format(args.weights, ap))

if __name__ == '__main__':
    args = parse_args()
    evaluate()
