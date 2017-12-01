import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

from coco_data_loader import CocoDataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from entity import JointType, params
from pose_detector import PoseDetector
from pose_detector import draw_person_pose


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='COCO evaluation')
    parser.add_argument('arch', choices=params['archs'].keys(), default='posenet', help='Model architecture')
    parser.add_argument('weights', help='weights file path')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--vis', action='store_true', help='visualize results')
    args = parser.parse_args()

    coco_val = COCO(os.path.join(params['coco_dir'], 'annotations/person_keypoints_val2017.json'))
    eval_loader = CocoDataLoader(coco_val, mode='eval', n_samples=None)

    pose_detector = PoseDetector(args.arch, args.weights, device=args.gpu)

    res = []
    imgIds = []
    # for i in range(len(eval_loader)):
    for i in range(100):
    # for i in [9]:
        print(i)
        img, annotations, img_id = eval_loader.get_example(i)
        if annotations is None:
            print('None')
            continue
        imgIds.append(img_id)

        person_pose_array = pose_detector(img)

        for person_pose in person_pose_array:
            res_dict = {}
            res_dict['category_id'] = 1
            res_dict['image_id'] = img_id
            res_dict['score'] = 1

            keypoints = np.zeros((len(params['coco_joint_indices']), 3))
            for joint, jt in zip(person_pose, JointType):
                if joint is not None and jt in params['coco_joint_indices']:
                    j = params['coco_joint_indices'].index(jt)
                    keypoints[j] = joint
            res_dict['keypoints'] = keypoints.ravel()
            res.append(res_dict)

        if args.vis:
            img = draw_person_pose(img, person_pose_array)

            # for ann in annotations:
            #     for joint in np.array(ann['keypoints']).reshape(-1, 3)[:, :2].astype('i'):
            #         cv2.circle(img, tuple(joint.tolist()), 3, (0, 0, 255), -1)

            cv2.imshow('results', img)
            cv2.waitKey(1)

        # # GT (test)
        # for ann in annotations:
        #     if ann['num_keypoints'] > 0:
        #         res_dict = {}
        #         res_dict['category_id'] = 1
        #         res_dict['image_id'] = img_id
        #         res_dict['score'] = 0
        #
        #         k = np.array(ann['keypoints']).reshape(17, 3)
        #         k[:, 0] += 1
        #         k[:, 1] += 1
        #         res_dict['keypoints'] = k.ravel()
        #
        #         # res_dict['keypoints'] = ann['keypoints']
        #         res.append(res_dict)

    cocoDt = coco_val.loadRes(res)
    cocoEval = COCOeval(coco_val, cocoDt, 'keypoints')
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
