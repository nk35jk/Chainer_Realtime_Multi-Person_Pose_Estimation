import os
import cv2
import copy
import json
import glob
import random
import argparse
import datetime
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import chainer
from chainer import cuda, training, reporter, function
from chainer.training import StandardUpdater, extensions
from chainer import serializers, optimizers, functions as F

from entity import JointType, params
from coco_data_loader import CocoDataLoader
from pose_detector import PoseDetector, draw_person_pose

from models import CocoPoseNet, posenet, nn1, resnetfpn, pspnet, student, cpn, mobilenet


class GradientScaling(object):

    name = 'GradientScaling'

    def __init__(self, layer_names, scale):
        self.layer_names = layer_names
        self.scale = scale

    def __call__(self, opt):
        for layer_name in self.layer_names:
            for param in opt.target[layer_name].params(False):
                grad = param.grad
                with cuda.get_device_from_array(grad):
                    grad *= self.scale


def compute_loss(imgs, pafs_ys, heatmaps_ys, pafs_t, heatmaps_t,
                 pafs_teacher, heatmaps_teacher, ignore_mask, device):
    xp = cuda.get_array_module(imgs)

    heatmap_loss_log = []
    paf_loss_log = []
    total_loss = 0

    paf_masks = ignore_mask[:, None].repeat(pafs_t.shape[1], axis=1)
    heatmap_masks = ignore_mask[:, None].repeat(heatmaps_t.shape[1], axis=1)

    # compute loss on each stage
    for pafs_y, heatmaps_y in zip(pafs_ys, heatmaps_ys):
        if not args.only_soft:
            stage_pafs_t = pafs_t.copy()
            stage_heatmaps_t = heatmaps_t.copy()
            stage_paf_masks = paf_masks.copy()
            stage_heatmap_masks = heatmap_masks.copy()

            if pafs_y.shape != stage_pafs_t.shape:
                stage_pafs_t = F.resize_images(stage_pafs_t, pafs_y.shape[2:]).data
                stage_heatmaps_t = F.resize_images(stage_heatmaps_t, pafs_y.shape[2:]).data
                stage_paf_masks = F.resize_images(stage_paf_masks.astype('f'), pafs_y.shape[2:]).data > 0
                stage_heatmap_masks = F.resize_images(stage_heatmap_masks.astype('f'), pafs_y.shape[2:]).data > 0

            if (args.distill or args.comp) and args.modify:
                # modify soft pafs
                paf_norm = (pafs_teacher[:, ::2]**2 + pafs_teacher[:, 1::2]**2)
                paf_norm_m = -(paf_norm - 1)**2 + 1
                multiplier = xp.where(paf_norm > 1e-3, paf_norm_m/paf_norm, 0)
                pafs_teacher = xp.repeat(multiplier, 2, axis=1) * pafs_teacher
                # modify soft heatmaps
                heatmaps_teacher_m = -(heatmaps_teacher - 1)**2 + 1
                heatmaps_teacher_m[:, -1] = heatmaps_teacher[:, -1]**2
                heatmaps_teacher = heatmaps_teacher_m

            if args.comp:
                """hard targetとsoft targetで絶対値が大きい方を学習ラベルとしても用いる"""
                pafs_t_mag = stage_pafs_t[:, ::2]**2 + stage_pafs_t[:, 1::2]**2
                pafs_t_mag = xp.repeat(pafs_t_mag, 2, axis=1)
                pafs_teacher_mag = pafs_teacher[:, ::2]**2 + pafs_teacher[:, 1::2]**2
                pafs_teacher_mag = xp.repeat(pafs_teacher_mag, 2, axis=1)
                stage_pafs_t[pafs_t_mag < pafs_teacher_mag] = pafs_teacher[pafs_t_mag < pafs_teacher_mag]

                stage_heatmaps_t[:, :-1][stage_heatmaps_t[:, :-1] < heatmaps_teacher[:, :-1]] = heatmaps_teacher[:, :-1][stage_heatmaps_t[:, :-1] < heatmaps_teacher[:, :-1]].copy()
                stage_heatmaps_t[:, -1][stage_heatmaps_t[:, -1] > heatmaps_teacher[:, -1]] = heatmaps_teacher[:, -1][stage_heatmaps_t[:, -1] > heatmaps_teacher[:, -1]].copy()

                # plt.imshow(stage_heatmaps_t[1, -1], vmin=0, vmax=1); plt.show()
                # plt.imshow(heatmaps_teacher[1, -1], vmin=0, vmax=1); plt.show()
                # plt.imshow(stage_heatmaps_t[1, -1], vmin=0, vmax=1); plt.show()
                #
                # plt.imshow(stage_pafs_t[1, 7], vmin=-1, vmax=1); plt.show()
                # plt.imshow(pafs_teacher[1, 7], vmin=-1, vmax=1); plt.show()
                # plt.imshow(stage_pafs_t[1, 7], vmin=-1, vmax=1); plt.show()

            stage_pafs_t[stage_paf_masks == True] = pafs_y.data[stage_paf_masks == True]
            stage_heatmaps_t[stage_heatmap_masks == True] = heatmaps_y.data[stage_heatmap_masks == True]

            pafs_loss = F.mean_squared_error(pafs_y, stage_pafs_t)
            heatmaps_loss = F.mean_squared_error(heatmaps_y, stage_heatmaps_t)

        if args.distill:
            soft_pafs_loss = F.mean_squared_error(pafs_y, pafs_teacher)
            soft_heatmaps_loss = F.mean_squared_error(heatmaps_y, heatmaps_teacher)

        if args.distill and args.only_soft:
            total_loss += soft_pafs_loss + soft_heatmaps_loss
        elif args.distill:
            total_loss += 0.5 * (pafs_loss + heatmaps_loss) + 0.5 * (soft_pafs_loss + soft_heatmaps_loss)
        elif not args.distill:
            total_loss += pafs_loss + heatmaps_loss

        if args.only_soft:
            paf_loss_log.append(float(cuda.to_cpu(soft_pafs_loss.data)))
            heatmap_loss_log.append(float(cuda.to_cpu(soft_heatmaps_loss.data)))
        else:
            paf_loss_log.append(float(cuda.to_cpu(pafs_loss.data)))
            heatmap_loss_log.append(float(cuda.to_cpu(heatmaps_loss.data)))

    return total_loss, paf_loss_log, heatmap_loss_log


def preprocess(imgs):
    xp = cuda.get_array_module(imgs)
    x_data = imgs.astype('f')
    if args.arch in ['posenet']:
        x_data /= 255
        x_data -= 0.5
    elif args.arch in ['nn1', 'resnetfpn', 'psp', 'student', 'cpn', 'mobilenet']:
        x_data -= xp.array([104, 117, 123])
    x_data = x_data.transpose(0, 3, 1, 2)
    return x_data


class Updater(StandardUpdater):

    def __init__(self, iterator, model, teacher, optimizer, device=None):
        super(Updater, self).__init__(iterator, optimizer, device=device)
        self.teacher = teacher

    def update_core(self):
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Update base network parameters
        if self.iteration == 2000:
            if args.arch == 'posenet':
                layer_names = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1',
                               'conv3_2', 'conv3_3', 'conv3_4', 'conv4_1', 'conv4_2']
                for layer_name in layer_names:
                    optimizer.target[layer_name].enable_update()
            elif args.arch in ['resnetfpn', 'pspnet', 'cpn']:
                optimizer.target.res.enable_update()
            elif args.arch in ['nn1', 'student']:
                model.squeeze.enable_update()

        if 100000 <= self.iteration < 200000:
            optimizer.alpha = 1e-5
        elif 200000 <= self.iteration:
            optimizer.alpha = 1e-6

        batch = train_iter.next()

        imgs, pafs, heatmaps, ignore_mask = self.converter(batch, self.device)

        x_data = preprocess(imgs)

        pafs_ys, heatmaps_ys = optimizer.target(x_data)

        pafs_teacher = heatmaps_teacher = None
        if self.teacher:
            x_data = ((imgs.astype('f') / 255) - 0.5).transpose(0, 3, 1, 2)
            h1s, h2s = self.teacher(x_data)
            pafs_teacher = h1s[-1].data
            heatmaps_teacher = h2s[-1].data

        loss, paf_loss_log, heatmap_loss_log = compute_loss(
            imgs, pafs_ys, heatmaps_ys, pafs, heatmaps, pafs_teacher,
            heatmaps_teacher, ignore_mask, self.device)

        reporter.report({
            'main/loss': loss,
            'main/paf': sum(paf_loss_log),
            'main/heat': sum(heatmap_loss_log),
        })

        optimizer.target.cleargrads()
        loss.backward()
        optimizer.update()


class Validator(extensions.Evaluator):

    def __init__(self, iterator, model, teacher, device=None):
        super(Validator, self).__init__(iterator, model, device=device)
        self.iterator = iterator
        self.teacher = teacher

    def evaluate(self):
        val_iter = self.get_iterator('main')
        model = self.get_target('main')

        it = copy.copy(val_iter)

        summary = reporter.DictSummary()
        res = []
        for i, batch in enumerate(it):
            observation = {}
            with reporter.report_scope(observation):
                imgs, pafs, heatmaps, ignore_mask = self.converter(batch, self.device)
                with function.no_backprop_mode():
                    x_data = preprocess(imgs)

                    pafs_ys, heatmaps_ys = model(x_data)

                    pafs_teacher = heatmaps_teacher = None
                    if self.teacher:
                        x_data = ((imgs.astype('f') / 255) - 0.5).transpose(0, 3, 1, 2)
                        h1s, h2s = self.teacher(x_data)
                        pafs_teacher = h1s[-1].data
                        heatmaps_teacher = h2s[-1].data

                    loss, paf_loss_log, heatmap_loss_log = compute_loss(
                        imgs, pafs_ys, heatmaps_ys, pafs, heatmaps, pafs_teacher,
                        heatmaps_teacher, ignore_mask, self.device)

                    observation['val/loss'] = cuda.to_cpu(loss.data)
                    observation['val/paf'] = sum(paf_loss_log)
                    observation['val/heat'] = sum(heatmap_loss_log)
            summary.add(observation)
        return summary.compute_mean()


class Evaluator(extensions.Evaluator):

    def __init__(self, cocoGt, iterator, model, device=None):
        super(Evaluator, self).__init__(iterator, model, device=device)
        self.cocoGt = cocoGt
        self.pose_detector = PoseDetector(model=model, device=device, precise=True)

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
                poses, scores = self.pose_detector(img)

                for pose, score in zip(poses, scores):
                    res_dict = {}
                    res_dict['category_id'] = 1
                    res_dict['image_id'] = img_id
                    res_dict['score'] = score * sum(pose[:, 2] > 0)

                    keypoints = np.zeros((len(params['coco_joint_indices']), 3))
                    for joint, jt in zip(pose, JointType):
                        if joint is not None and jt in params['coco_joint_indices']:
                            i = params['coco_joint_indices'].index(jt)
                            keypoints[i] = joint
                    res_dict['keypoints'] = keypoints.ravel()
                    res.append(res_dict)

            img = draw_person_pose(img, poses)

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


def parse_args():
    parser = argparse.ArgumentParser(description='Train pose estimation')
    parser.add_argument('--arch', '-a', choices=params['archs'].keys(), default='posenet',
                        help='Model architecture')
    parser.add_argument('--batchsize', '-B', type=int, default=10,
                        help='Learning minibatch size')
    parser.add_argument('--valbatchsize', '-b', type=int, default=4,
                        help='Validation minibatch size')
    parser.add_argument('--val_samples', type=int, default=100,
                        help='Number of validation samples')
    parser.add_argument('--eval_samples', type=int, default=100,
                        help='Number of validation samples')
    parser.add_argument('--iteration', '-i', type=int, default=300000,
                        help='Number of iterations to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU')
    parser.add_argument('--initmodel',
                        help='Initialize the model from given file')
    parser.add_argument('--loaderjob', '-j', type=int,
                        help='Number of parallel data loading processes')
    parser.add_argument('--resume', '-r', default='',
                        help='Initialize the trainer from given file')
    parser.add_argument('--out', '-o', default='result/test',
                        help='Output directory')
    parser.add_argument('--stages', '-s', type=int, default=6,
                        help='number of posenet stages')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--distill', action='store_true')
    parser.add_argument('--only_soft', action='store_true'
                        help='Train student model with only soft target')
    parser.add_argument('--comp', action='store_true'
                        help='Complete label with output of teacher model')
    parser.add_argument('--modify', action='store_true'
                        help='Modify soft target')
    parser.set_defaults(test=False)
    parser.set_defaults(distill=False)
    parser.set_defaults(only_soft=False)
    parser.set_defaults(comp=False)
    parser.set_defaults(modify=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    np.random.seed(0)
    random.seed(0)

    # Prepare model
    if args.arch == 'posenet':
        model = params['archs'][args.arch](stages=args.stages)
    else:
        model = params['archs'][args.arch]()

    if args.arch == 'posenet':
        posenet.copy_vgg_params(model)
    elif args.arch in ['nn1', 'student']:
        nn1.copy_squeezenet_params(model.squeeze)
    elif args.arch in ['resnetfpn', 'pspnet', 'cpn']:
        chainer.serializers.load_npz('models/resnet50.npz', model.res)

    if args.initmodel:
        print('Load model from', args.initmodel)
        chainer.serializers.load_npz(args.initmodel, model)

    # Prepare teacher model for distillation
    teacher = None
    if args.distill or args.comp:
        teacher = posenet.PoseNet()
        serializers.load_npz('models/posenet_190k_ap_0.544.npz', teacher)
        teacher.disable_update()

    # Set up GPU
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
        if args.distill or args.comp:
            teacher.to_gpu()

    # Set up an optimizer
    # optimizer = optimizers.MomentumSGD(lr=1e-3, momentum=0.9)
    optimizer = optimizers.Adam(alpha=1e-4, beta1=0.9, beta2=0.999, eps=1e-08)
    optimizer.setup(model)
    # optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))
    if args.arch == 'posenet':
        layer_names = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1',
                       'conv3_2', 'conv3_3', 'conv3_4', 'conv4_1', 'conv4_2',
                       'conv4_3_CPM', 'conv4_4_CPM']
        optimizer.add_hook(GradientScaling(layer_names, 1/4))

    # Fix base network parameters
    if not args.resume:
        if args.arch == 'posenet':
            layer_names = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1',
                           'conv3_2', 'conv3_3', 'conv3_4', 'conv4_1', 'conv4_2']
            for layer_name in layer_names:
                model[layer_name].disable_update()
        elif args.arch in ['resnetfpn', 'pspnet', 'cpn']:
            model.res.disable_update()
        elif args.arch in ['nn1', 'student']:
            model.squeeze.disable_update()

    # Load datasets
    coco_train = COCO(os.path.join(params['coco_dir'], 'annotations/person_keypoints_train2017.json'))
    coco_val = COCO(os.path.join(params['coco_dir'], 'annotations/person_keypoints_val2017.json'))
    train_loader = CocoDataLoader(coco_train, model.insize, mode='train')
    val_loader = CocoDataLoader(coco_val, model.insize, mode='val', n_samples=args.val_samples)
    # eval_loader = CocoDataLoader(coco_val, model, mode='eval', n_samples=args.eval_samples)

    # Set up iterators
    if args.loaderjob:
        multiprocessing.set_start_method('spawn')  # to avoid MultiprocessIterator's bug
        train_iter = chainer.iterators.MultiprocessIterator(
            train_loader, args.batchsize, n_processes=args.loaderjob)
        val_iter = chainer.iterators.MultiprocessIterator(
            val_loader, args.valbatchsize, n_processes=args.loaderjob, repeat=False, shuffle=False)
        # eval_iter = chainer.iterators.MultiprocessIterator(
        #     eval_loader, 1, n_processes=args.loaderjob, repeat=False, shuffle=False, shared_mem=None)
    else:
        train_iter = chainer.iterators.SerialIterator(train_loader, args.batchsize)
        val_iter = chainer.iterators.SerialIterator(
            val_loader, args.valbatchsize, repeat=False, shuffle=False)
        # eval_iter = chainer.iterators.SerialIterator(
        #     eval_loader, 1, repeat=False, shuffle=False)

    # Set up a trainer
    updater = Updater(train_iter, model, teacher, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.iteration, 'iteration'), args.out)

    val_interval = (10 if args.test else 1000), 'iteration'
    log_interval = (1 if args.test else 20), 'iteration'

    trainer.extend(Validator(val_iter, model, teacher, device=args.gpu),
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
        'main/heat', 'val/heat',#'AP', 'AR'
    ]), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=1))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    # Save training parameters
    if not os.path.exists(args.out):
        os.makedirs(args.out)
    txt = '@{}'.format(datetime.datetime.now().strftime('%y%m%d_%H%M'))
    with open(os.path.join(args.out, txt), 'w') as f:
        pass
    with open(os.path.join(args.out, 'params.json'), 'w') as f:
        json.dump(vars(args), f)

    trainer.run()
