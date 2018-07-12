import os
import sys
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
    parser.add_argument('arch', choices=params['archs'].keys(),
                        default='posenet', help='Model architecture')
    parser.add_argument('weights', help='weights file path')
    parser.add_argument('--stages', '-s', type=int, default=6,
                        help='number of stages of posenet')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--display', action='store_true',
                        help='visualize inference results')
    args = parser.parse_args()
    params['inference_img_size'] = params['archs'][args.arch].insize
    params['downscale'] = params['archs'][args.arch].downscale
    params['pad'] = params['archs'][args.arch].pad
    return args


def evaluate():
    pose_detector = PoseDetector(args.arch, args.weights, device=args.gpu,
                                 precise=True, stages=args.stages)

    coco_val = COCO(os.path.join(params['coco_dir'],
                                 'annotations/person_keypoints_val2017.json'))
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
            res_dict['keypoints'] = keypoints.ravel()
            res.append(res_dict)

        if args.display:
            img = draw_person_pose(img, poses)
            cv2.imshow('results', img)
            k = cv2.waitKey(1)
            if k == ord('q'):
                sys.exit()

    cocoDt = coco_val.loadRes(res)
    cocoEval = COCOeval(coco_val, cocoDt, 'keypoints')
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    ap = cocoEval.stats[0]

    path = '{}/evaluation_results.txt'.format('/'.join(args.weights.split('/')[:-1]))
    with open(path, 'a') as f:
        f.write('{}, {}\n'.format(args.weights, ap))

if __name__ == '__main__':
    args = parse_args()
    evaluate()
