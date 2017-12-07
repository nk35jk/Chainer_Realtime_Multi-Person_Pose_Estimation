import os
import cv2
import copy
import json
import glob
import random
import numpy as np
import matplotlib.pyplot as plt

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import chainer
from chainer import cuda, training, reporter, function
from chainer.training import StandardUpdater, extensions
from chainer import serializers, optimizers, functions as F

from entity import JointType, params, parse_args
from coco_data_loader import CocoDataLoader
from pose_detector import PoseDetector, draw_person_pose

from models import CocoPoseNet
from models import nn1
from models import resnetfpn


def compute_loss(imgs, pafs_ys, heatmaps_ys, masks_ys, pafs_t, heatmaps_t, ignore_mask, stuff_mask, compute_mask, device):
    heatmap_loss_log = []
    paf_loss_log = []
    mask_loss_log = []
    total_loss = 0

    paf_masks = ignore_mask[:, None].repeat(pafs_t.shape[1], axis=1)
    heatmap_masks = ignore_mask[:, None].repeat(heatmaps_t.shape[1], axis=1)

    for pafs_y, heatmaps_y, masks_y in zip(pafs_ys, heatmaps_ys, masks_ys): # compute loss on each stage
        stage_pafs_t = pafs_t.copy()
        stage_heatmaps_t = heatmaps_t.copy()
        stage_paf_masks = paf_masks.copy()
        stage_heatmap_masks = heatmap_masks.copy()
        stage_stuff_mask = stuff_mask.copy()

        if pafs_y.shape != stage_pafs_t.shape:
            stage_pafs_t = F.resize_images(stage_pafs_t, pafs_y.shape[2:]).data
            stage_heatmaps_t = F.resize_images(stage_heatmaps_t, pafs_y.shape[2:]).data
            stage_paf_masks = F.resize_images(stage_paf_masks.astype('f'), pafs_y.shape[2:]).data > 0
            stage_heatmap_masks = F.resize_images(stage_heatmap_masks.astype('f'), pafs_y.shape[2:]).data > 0

            if device >= 0:
                stage_stuff_mask = cuda.to_cpu(stage_stuff_mask)
            stage_stuff_mask = cv2.resize(stage_stuff_mask.transpose(1, 2, 0),
                pafs_y.shape[2:], interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)
            if device >= 0:
                stage_stuff_mask = cuda.to_gpu(stage_stuff_mask)

        stage_pafs_t[stage_paf_masks == True] = pafs_y.data[stage_paf_masks == True]
        stage_heatmaps_t[stage_heatmap_masks == True] = heatmaps_y.data[stage_heatmap_masks == True]

        pafs_loss = F.mean_squared_error(pafs_y, stage_pafs_t)
        heatmaps_loss = F.mean_squared_error(heatmaps_y, stage_heatmaps_t)
        mask_loss = 0
        if compute_mask:
            mask_loss = F.softmax_cross_entropy(masks_y, stage_stuff_mask)
        total_loss += pafs_loss + heatmaps_loss + 0.01 * mask_loss

        paf_loss_log.append(float(cuda.to_cpu(pafs_loss.data)))
        heatmap_loss_log.append(float(cuda.to_cpu(heatmaps_loss.data)))
        if type(mask_loss) == int:
            mask_loss_log.append(float(mask_loss))
        else:
            mask_loss_log.append(float(cuda.to_cpu(mask_loss.data)))

    return total_loss, paf_loss_log, heatmap_loss_log, mask_loss_log


def preprocess(imgs):
    xp = cuda.get_array_module(imgs)
    x_data = imgs.astype('f')
    if args.arch in ['posenet']:
        x_data /= 255
        x_data -= 0.5
    elif args.arch in ['nn1', 'resnetfpn']:
        x_data -= xp.array([104, 117, 123])
    x_data = x_data.transpose(0, 3, 1, 2)
    return x_data


class Updater(StandardUpdater):

    def __init__(self, iterator, model, optimizer, compute_mask, device=None):
        super(Updater, self).__init__(iterator, optimizer, device=device)
        self.compute_mask = compute_mask

    def update_core(self):
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        batch = train_iter.next()
        imgs, pafs, heatmaps, ignore_mask, stuff_mask = self.converter(batch, self.device)

        x_data = preprocess(imgs)

        if self.compute_mask:
            pafs_ys, heatmaps_ys, masks_ys = optimizer.target(x_data)
        else:
            pafs_ys, heatmaps_ys = optimizer.target(x_data)
            masks_ys = [None] * len(pafs_ys)

        loss, paf_loss_log, heatmap_loss_log, mask_loss_log = compute_loss(
            imgs, pafs_ys, heatmaps_ys, masks_ys, pafs, heatmaps, ignore_mask, stuff_mask, self.compute_mask, self.device)

        reporter.report({
            'main/loss': loss,
            'main/paf': sum(paf_loss_log),
            'main/heatmap': sum(heatmap_loss_log),
            'main/mask': sum(mask_loss_log),
        })

        optimizer.target.cleargrads()
        loss.backward()
        optimizer.update()


class Validator(extensions.Evaluator):

    def __init__(self, iterator, model, compute_mask, device=None):
        super(Validator, self).__init__(iterator, model, device=device)
        self.iterator = iterator
        self.compute_mask = compute_mask

    def evaluate(self):
        val_iter = self.get_iterator('main')
        model = self.get_target('main')

        it = copy.copy(val_iter)

        summary = reporter.DictSummary()
        res = []
        for i, batch in enumerate(it):
            observation = {}
            with reporter.report_scope(observation):
                imgs, pafs, heatmaps, ignore_mask, stuff_mask = self.converter(batch, self.device)
                with function.no_backprop_mode():
                    x_data = preprocess(imgs)

                    if self.compute_mask:
                        pafs_ys, heatmaps_ys, masks_ys = model(x_data)
                    else:
                        pafs_ys, heatmaps_ys = model(x_data)
                        masks_ys = [None] * len(pafs_ys)

                    loss, paf_loss_log, heatmap_loss_log, mask_loss_log = compute_loss(
                        imgs, pafs_ys, heatmaps_ys, masks_ys, pafs, heatmaps, ignore_mask, stuff_mask, self.compute_mask, self.device)
                    observation['val/loss'] = cuda.to_cpu(loss.data)
                    observation['val/paf'] = sum(paf_loss_log)
                    observation['val/heatmap'] = sum(heatmap_loss_log)
                    observation['val/mask'] = sum(mask_loss_log)
            summary.add(observation)
        return summary.compute_mean()


class Evaluator(extensions.Evaluator):

    def __init__(self, cocoGt, iterator, model, device=None):
        super(Evaluator, self).__init__(iterator, model, device=device)
        self.cocoGt = cocoGt
        self.pose_detector = PoseDetector(model=model, device=device)

    def evaluate(self):
        val_iter = self.get_iterator('main')
        model = self.get_target('main')
        self.pose_detector.model = model

        it = copy.copy(val_iter)

        res = []
        imgIds = []
        for batch in it:
            img, annotation, img_id = batch[0]
            with function.no_backprop_mode():
                imgIds.append(img_id)
                person_pose_array = self.pose_detector(img)

                for person_pose in person_pose_array:
                    res_dict = {}
                    res_dict['category_id'] = 1
                    res_dict['image_id'] = img_id
                    res_dict['score'] = 1

                    keypoints = np.zeros((len(params['coco_joint_indices']), 3))
                    for joint, jt in zip(person_pose, JointType):
                        if joint is not None and jt in params['coco_joint_indices']:
                            i = params['coco_joint_indices'].index(jt)
                            keypoints[i] = joint
                    res_dict['keypoints'] = keypoints.ravel()
                    res.append(res_dict)

            img = draw_person_pose(img, person_pose_array)

            # import matplotlib.pyplot as plt
            # plt.imshow(img[..., ::-1]); plt.show()
            # cv2.imwrite('result/{}.jpg'.format(img_id), img)

        summary = reporter.DictSummary()
        stat_names = ['AP', 'AP50', 'AP75', 'AP_M', 'AP_L', 'AR', 'AR50', 'AR75', 'AR_M', 'AR_L']
        if len(res) > 0:
            cocoDt = self.cocoGt.loadRes(res)
            cocoEval = COCOeval(self.cocoGt, cocoDt, 'keypoints')
            cocoEval.params.imgIds = imgIds
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()

            observation = dict(zip(stat_names, cocoEval.stats))
        else:
            observation = dict(zip(stat_names, [0]*len(stat_names)))
        summary.add(observation)
        return summary.compute_mean()


if __name__ == '__main__':
    args = parse_args()

    model = params['archs'][args.arch](compute_mask=args.mask)

    if args.arch == 'posenet':
        CocoPoseNet.copy_vgg_params(model)
    elif args.arch == 'nn1':
        nn1.copy_squeezenet_params(model.squeeze)
    elif args.arch == 'resnetfpn':
        chainer.serializers.load_npz('models/resnet50.npz', model.res)

    if args.initmodel:
        print('Load model from', args.initmodel)
        chainer.serializers.load_npz(args.initmodel, model)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # Load the datasets
    coco_train = COCO(os.path.join(params['coco_dir'], 'annotations/person_keypoints_train2017.json'))
    coco_val = COCO(os.path.join(params['coco_dir'], 'annotations/person_keypoints_val2017.json'))
    train_loader = CocoDataLoader(coco_train, mode='train')
    val_loader = CocoDataLoader(coco_val, mode='val', n_samples=args.val_samples)
    eval_loader = CocoDataLoader(coco_val, mode='eval', n_samples=args.eval_samples)

    if args.loaderjob:
        train_iter = chainer.iterators.MultiprocessIterator(
            train_loader, args.batchsize, n_processes=args.loaderjob)
        val_iter = chainer.iterators.MultiprocessIterator(
            val_loader, args.valbatchsize, n_processes=args.loaderjob, repeat=False, shuffle=False)
        eval_iter = chainer.iterators.MultiprocessIterator(
            eval_loader, 1, n_processes=args.loaderjob, repeat=False, shuffle=False)
    else:
        train_iter = chainer.iterators.SerialIterator(train_loader, args.batchsize)
        val_iter = chainer.iterators.SerialIterator(
            val_loader, args.valbatchsize, repeat=False, shuffle=False)
        eval_iter = chainer.iterators.SerialIterator(
            eval_loader, 1, repeat=False, shuffle=False)

    # Set up an optimizer
    # optimizer = optimizers.MomentumSGD(lr=4e-5, momentum=0.9)
    optimizer = optimizers.Adam(alpha=1e-4, beta1=0.9, beta2=0.999, eps=1e-08)
    optimizer.setup(model)
    # optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))

    # Set up a trainer
    updater = Updater(train_iter, model, optimizer, args.mask, device=args.gpu)
    trainer = training.Trainer(updater, (args.iteration, 'iteration'), args.out)

    val_interval = (10 if args.test else 1000), 'iteration'
    log_interval = (1 if args.test else 10), 'iteration'

    trainer.extend(Validator(val_iter, model, args.mask, device=args.gpu),
                   trigger=val_interval)
    # trainer.extend(Evaluator(coco_val, eval_iter, model, device=args.gpu),
    #                trigger=val_interval)
    # trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=val_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}'), trigger=val_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'main/loss', 'val/loss', 'main/paf', 'val/paf',
        'main/heatmap', 'val/heatmap', 'main/mask', 'val/mask',
    ]), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=1))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()
