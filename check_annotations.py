import os
import sys
import cv2
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from chainer.dataset import DatasetMixin
from pycocotools.coco import COCO

from entity import JointType, params
from coco_data_loader import CocoDataLoader


if __name__ == '__main__':
    mode = 'train'  # train, val
    coco = COCO(os.path.join(params['coco_dir'], 'annotations/person_keypoints_{}2017.json'.format(mode)))
    data_loader = CocoDataLoader(params['coco_dir'], coco, params['insize'],
                                 mode=mode, use_all_images=False, use_ignore_mask=True,
                                 augment_data=False, resize_data=False)

    person_cnt = 0
    areas = []
    lengths = []
    ratios = []
    n_kpts = []

    for i in range(len(data_loader)):
        img_id = data_loader.imgIds[i]
        print('{}, img_id: {}'.format(i, img_id))
        anno_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)

        if len(anno_ids) > 0:
            annotations = coco.loadAnns(anno_ids)

            for ann in annotations:
                if ann['category_id'] == 1 and ann['num_keypoints'] >= 1:
                    person_cnt += 1
                    areas.append(ann['area'])
                    x, y, w, h = ann['bbox']
                    length = (w**2 + h**2)**0.5
                    lengths.append(length)
                    ratios.append(ann['area']**0.5 / length)
                    n_kpts.append(ann['num_keypoints'])
                    print(ann['area'], length)
