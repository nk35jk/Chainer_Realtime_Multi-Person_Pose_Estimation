import os
import cv2
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from coco_data_loader import CocoDataLoader
from pycocotools.coco import COCO

from entity import JointType, params

import chainer
from chainer import serializers, cuda, functions as F

chainer.config.enable_backprop = False
chainer.config.train = False


def compute_loss(imgs, pafs_ys, heatmaps_ys, pafs_t, heatmaps_t, ignore_mask, device):
    paf_masks = ignore_mask[:, None].repeat(pafs_t.shape[1], axis=1)
    heatmap_masks = ignore_mask[:, None].repeat(heatmaps_t.shape[1], axis=1)

    pafs_loss = heatmaps_loss = 0

    for pafs_y, heatmaps_y in zip(pafs_ys, heatmaps_ys):
        stage_pafs_t = pafs_t.copy()
        stage_heatmaps_t = heatmaps_t.copy()
        stage_paf_masks = paf_masks.copy()
        stage_heatmap_masks = heatmap_masks.copy()

        if pafs_y.shape != stage_pafs_t.shape:
            stage_pafs_t = F.resize_images(stage_pafs_t, pafs_y.shape[2:]).data
            stage_heatmaps_t = F.resize_images(stage_heatmaps_t, pafs_y.shape[2:]).data
            stage_paf_masks = F.resize_images(stage_paf_masks.astype('f'), pafs_y.shape[2:]).data > 0
            stage_heatmap_masks = F.resize_images(stage_heatmap_masks.astype('f'), pafs_y.shape[2:]).data > 0

        stage_pafs_t[stage_paf_masks == True] = pafs_y.data[stage_paf_masks == True]
        stage_heatmaps_t[stage_heatmap_masks == True] = heatmaps_y.data[stage_heatmap_masks == True]

        pafs_loss += F.mean_squared_error(pafs_y, stage_pafs_t)
        heatmaps_loss += F.mean_squared_error(heatmaps_y, stage_heatmaps_t)

    if device >= 0:
        pafs_loss = pafs_loss.get()
        heatmaps_loss = heatmaps_loss.get()

    return pafs_loss.data, heatmaps_loss.data


def preprocess(imgs):
    xp = cuda.get_array_module(imgs)
    x_data = imgs.astype('f')
    x_data /= 255
    x_data -= 0.5
    x_data = x_data.transpose(0, 3, 1, 2)
    return x_data


def parse_args():
    parser = argparse.ArgumentParser(description='COCO evaluation')
    parser.add_argument('arch', choices=params['archs'].keys(), default='posenet', help='Model architecture')
    parser.add_argument('weights', help='weights file path')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--display', action='store_true', help='visualize results')
    parser.add_argument('--stages', '-s', type=int, default=6, help='number of posenet stages')
    args = parser.parse_args()
    params['inference_img_size'] = params['archs'][args.arch].insize
    params['downscale'] = params['archs'][args.arch].downscale
    params['pad'] = params['archs'][args.arch].pad
    return args


def list_loss():
    if args.arch == 'posenet':
        model = params['archs'][args.arch](stages=args.stages)
    else:
        model = params['archs'][args.arch]()

    if args.weights:
        serializers.load_npz(args.weights, model)

    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    coco_val = COCO(os.path.join(params['coco_dir'], 'annotations/person_keypoints_val2017.json'))
    val_loader = CocoDataLoader(coco_val, model.insize, mode='val', n_samples=None)

    pafs_losses = []
    heatmaps_losses = []

    for i in range(len(val_loader)):
    # for i in range(4):
        print('\r{:4d}'.format(i), end='')

        img, pafs, heatmaps, ignore_mask = val_loader.get_example(i)
        img = img[None]
        pafs_t = pafs[None]
        heatmaps_t = heatmaps[None]
        ignore_mask = ignore_mask[None]

        x_data = preprocess(img)

        if args.gpu >= 0:
            x_data = cuda.to_gpu(x_data)
            pafs_t = cuda.to_gpu(pafs_t)
            heatmaps_t = cuda.to_gpu(heatmaps_t)
            ignore_mask = cuda.to_gpu(ignore_mask)

        pafs_ys, heatmaps_ys = model(x_data)

        pafs_loss, heatmaps_loss = compute_loss(img, pafs_ys, heatmaps_ys, pafs_t, heatmaps_t, ignore_mask, args.gpu)

        pafs_losses.append(pafs_loss)
        heatmaps_losses.append(heatmaps_loss)

        # print('{} {}'.format(pafs_loss, heatmaps_loss))

        # if args.display:
        #     cv2.imwrite('result/img/result.png'.format(), img)
        #
        #     cv2.imshow('results', img)
        #     k = cv2.waitKey(1)
        #     if k == ord('q'):
        #         exit()

    plt.hist(pafs_losses)
    plt.xlabel('PAFs loss')
    plt.ylabel('Frequency')
    plt.savefig('result/pafs_loss_hist.png')
    plt.clf()

    plt.hist(heatmaps_losses)
    plt.xlabel('Heatmaps loss')
    plt.ylabel('Frequency')
    plt.savefig('result/heatmaps_loss_hist.png')
    plt.clf()


if __name__ == '__main__':
    args = parse_args()
    list_loss()
