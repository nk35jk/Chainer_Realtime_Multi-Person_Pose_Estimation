import os
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from entity import params


if __name__ == '__main__':
    coco_val = COCO(os.path.join(params['coco_dir'],
                                 'annotations/person_keypoints_val2017.json'))

    res_file1 = json.load(open('result/old/posenet171230/res1000.json'))
    res_file2 = json.load(open('result/old/posenet_label_comp/res1000.json'))
    imgIds1 = list(set([x['image_id'] for x in res_file1]))
    imgIds2 = list(set([x['image_id'] for x in res_file2]))
    imgIds = list(set(imgIds1) & set(imgIds2))

    cocoDt1 = coco_val.loadRes(res_file1)
    cocoDt2 = coco_val.loadRes(res_file2)
    cocoEval1 = COCOeval(coco_val, cocoDt1, 'keypoints')
    cocoEval2 = COCOeval(coco_val, cocoDt2, 'keypoints')
    aps1, aps2 = [], []
    for i in range(len(imgIds)):
        cocoEval1.params.imgIds = imgIds[i]
        cocoEval1.evaluate()
        cocoEval1.accumulate()
        cocoEval1.summarize()
        ap1 = cocoEval1.stats[0]
        aps1.append(ap1)

        cocoEval2.params.imgIds = imgIds[i]
        cocoEval2.evaluate()
        cocoEval2.accumulate()
        cocoEval2.summarize()
        ap2 = cocoEval2.stats[0]
        aps2.append(ap2)

    diff = np.array(aps2) - np.array(aps1)

    plt.hist(diff, bins=20)
    plt.grid()
    plt.xlim(-1, 1)
    plt.xlabel('AP difference')
    plt.ylabel('frequency')
    plt.savefig('result/ap_diff_hist.png', bbox_inches='tight')
    plt.clf()
