import os
import cv2
import copy
import json
import glob
import random
import argparse
import datetime
import subprocess
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
        if args.distill:
            stage_pafs_teacher_distill = pafs_teacher.copy()
            stage_heatmaps_teacher_distill = heatmaps_teacher.copy()

        if args.comp_paf:
            stage_pafs_teacher_comp = pafs_teacher.copy()
        if args.comp_heat:
            stage_heatmaps_teacher_comp = heatmaps_teacher.copy()

        if args.distill and args.modify:
            # modify soft pafs
            paf_norm = (stage_pafs_teacher_distill[:, ::2]**2 + stage_pafs_teacher_distill[:, 1::2]**2)
            paf_norm_m = -(paf_norm - 1)**2 + 1
            multiplier = xp.where(paf_norm > 1e-3, paf_norm_m/paf_norm, 0)
            stage_pafs_teacher_distill = xp.repeat(multiplier, 2, axis=1) * stage_pafs_teacher_distill
            # modify soft heatmaps
            heatmaps_teacher_m = -(stage_heatmaps_teacher_distill - 1)**2 + 1
            heatmaps_teacher_m[:, -1] = stage_heatmaps_teacher_distill[:, -1]**2
            stage_heatmaps_teacher_distill = heatmaps_teacher_m

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

            if args.comp_paf:
                """pafsを補正"""
                pafs_t_mag = stage_pafs_t[:, ::2]**2 + stage_pafs_t[:, 1::2]**2
                pafs_t_mag = xp.repeat(pafs_t_mag, 2, axis=1)
                pafs_teacher_mag = stage_pafs_teacher_comp[:, ::2]**2 + stage_pafs_teacher_comp[:, 1::2]**2
                pafs_teacher_mag = xp.repeat(pafs_teacher_mag, 2, axis=1)
                stage_pafs_t[pafs_t_mag < pafs_teacher_mag] = stage_pafs_teacher_comp[pafs_t_mag < pafs_teacher_mag]

            if args.comp_heat:
                """heatmapsを補正"""
                stage_heatmaps_t[:, :-1][stage_heatmaps_t[:, :-1] < stage_heatmaps_teacher_comp[:, :-1]] = stage_heatmaps_teacher_comp[:, :-1][stage_heatmaps_t[:, :-1] < stage_heatmaps_teacher_comp[:, :-1]].copy()
                stage_heatmaps_t[:, -1][stage_heatmaps_t[:, -1] > stage_heatmaps_teacher_comp[:, -1]] = stage_heatmaps_teacher_comp[:, -1][stage_heatmaps_t[:, -1] > stage_heatmaps_teacher_comp[:, -1]].copy()

            stage_pafs_t[stage_paf_masks == True] = pafs_y.data[stage_paf_masks == True]
            stage_heatmaps_t[stage_heatmap_masks == True] = heatmaps_y.data[stage_heatmap_masks == True]

            pafs_loss = F.mean_squared_error(pafs_y, stage_pafs_t)
            heatmaps_loss = F.mean_squared_error(heatmaps_y, stage_heatmaps_t)

        if args.distill:
            soft_pafs_loss = F.mean_squared_error(pafs_y, stage_pafs_teacher_distill)
            soft_heatmaps_loss = F.mean_squared_error(heatmaps_y, stage_heatmaps_teacher_distill)

        if args.distill and args.only_soft:
            total_loss += soft_pafs_loss + soft_heatmaps_loss
        elif args.distill:
            total_loss += 0.5 * (pafs_loss + heatmaps_loss) + 0.5 * (soft_pafs_loss + soft_heatmaps_loss)
        elif not args.distill:
            total_loss += pafs_loss + heatmaps_loss

        if args.distill and args.only_soft:
            paf_loss_log.append(soft_pafs_loss.data)
            heatmap_loss_log.append(soft_heatmaps_loss.data)
        elif args.distill:
            paf_loss_log.append(0.5*pafs_loss.data + 0.5*soft_pafs_loss.data)
            heatmap_loss_log.append(0.5*heatmaps_loss.data + 0.5*soft_heatmaps_loss.data)
        elif not args.distill:
            paf_loss_log.append(pafs_loss.data)
            heatmap_loss_log.append(heatmaps_loss.data)

    return total_loss, paf_loss_log, heatmap_loss_log


def preprocess(imgs):
    xp = cuda.get_array_module(imgs)
    x_data = imgs.astype('f')
    if args.arch in ['posenet', 'student']:
        x_data /= 255
        x_data -= 0.5
    else:
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
        if self.iteration == 4000:
            if args.arch == 'posenet':
                layer_names = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1',
                               'conv3_2', 'conv3_3', 'conv3_4', 'conv4_1', 'conv4_2']
                for layer_name in layer_names:
                    optimizer.target[layer_name].enable_update()
            elif args.arch in ['resnetfpn', 'pspnet', 'cpn',
                               'resnet50', 'resnet101', 'resnet152',
                               'resnet50-dilate', 'resnet101-dilate', 'resnet152-dilate']:
                optimizer.target.res.enable_update()
            elif args.arch in ['nn1']:
                optimizer.target.squeeze.enable_update()

        batch = train_iter.next()

        imgs, pafs, heatmaps, ignore_mask = self.converter(batch, self.device)

        xp = cuda.get_array_module(imgs)

        x_data = preprocess(imgs)

        pafs_ys, heatmaps_ys = optimizer.target(x_data)

        pafs_teacher = heatmaps_teacher = 0
        if self.teacher:
            if args.teacher_type == 'single':
                # single scale prediction
                x_data = ((imgs.astype('f') / 255) - 0.5).transpose(0, 3, 1, 2)
                with function.no_backprop_mode():
                    h1s, h2s = self.teacher(x_data)
                pafs_teacher += h1s[-1].data
                heatmaps_teacher += h2s[-1].data
                pafs_teacher_s = pafs_teacher.copy()
                heatmaps_teacher_s = heatmaps_teacher.copy()

            elif args.teacher_type == 'multi_avg':
                # multi scale and flip prediction (average)
                for scale in params['teacher_scales']:
                    insize = int(self.teacher.insize * scale)
                    outsize = self.teacher.insize // self.teacher.downscale
                    x_data = ((imgs.astype('f') / 255) - 0.5).transpose(0, 3, 1, 2)
                    x_data = F.resize_images(x_data, (insize, insize))

                    with function.no_backprop_mode():
                        h1s, h2s = self.teacher(x_data)
                    pafs_teacher += F.resize_images(h1s[-1], (outsize, outsize)).data
                    heatmaps_teacher += F.resize_images(h2s[-1], (outsize, outsize)).data
                pafs_teacher /= len(params['teacher_scales'])
                heatmaps_teacher /= len(params['teacher_scales'])
                pafs_teacher_a = pafs_teacher.copy()
                heatmaps_teacher_a = heatmaps_teacher.copy()

            elif args.teacher_type == 'multi_max':
                # multi scale and flip prediction (max)
                for scale in params['teacher_scales']:
                    insize = int(self.teacher.insize * scale)
                    outsize = self.teacher.insize // self.teacher.downscale
                    x_data = ((imgs.astype('f') / 255) - 0.5).transpose(0, 3, 1, 2)
                    x_data = F.resize_images(x_data, (insize, insize))

                    with function.no_backprop_mode():
                        h1s, h2s = self.teacher(x_data)

                    if pafs_teacher is 0 and heatmaps_teacher is 0:
                        pafs_teacher = F.resize_images(h1s[-1], (outsize, outsize)).data
                        heatmaps_teacher = F.resize_images(h2s[-1], (outsize, outsize)).data
                    else:
                        pafs_teacher2 = F.resize_images(h1s[-1], (outsize, outsize)).data
                        pafs_mag1 = pafs_teacher[:, ::2]**2 + pafs_teacher[:, 1::2]**2
                        pafs_mag1 = xp.repeat(pafs_mag1, 2, axis=1)
                        pafs_mag2 = pafs_teacher2[:, ::2]**2 + pafs_teacher2[:, 1::2]**2
                        pafs_mag2 = xp.repeat(pafs_mag2, 2, axis=1)
                        pafs_teacher[pafs_mag1 < pafs_mag2] = pafs_teacher2[pafs_mag1 < pafs_mag2]

                        heatmaps_teacher2 = F.resize_images(h2s[-1], (outsize, outsize)).data
                        heatmaps_teacher[:, :-1][heatmaps_teacher[:, :-1] < heatmaps_teacher2[:, :-1]] = heatmaps_teacher2[:, :-1][heatmaps_teacher[:, :-1] < heatmaps_teacher2[:, :-1]].copy()
                        heatmaps_teacher[:, -1][heatmaps_teacher[:, -1] > heatmaps_teacher2[:, -1]] = heatmaps_teacher2[:, -1][heatmaps_teacher[:, -1] > heatmaps_teacher2[:, -1]].copy()

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

    parser.add_argument('--arch', '-a', choices=params['archs'].keys(),
                        default='posenet',
                        help='model architecture')
    parser.add_argument('--stages', '-s', type=int, default=6,
                        help='number of posenet stages')
    parser.add_argument('--initmodel',
                        help='initialize the model from given file')

    parser.add_argument('--batchsize', '-B', type=int, default=10,
                        help='learning minibatch size')
    parser.add_argument('--valbatchsize', '-b', type=int, default=4,
                        help='validation minibatch size')
    parser.add_argument('--iteration', '-i', type=int, default=300000,
                        help='number of iterations to train')
    parser.add_argument('--loaderjob', '-j', type=int,
                        help='number of parallel data loading processes')
    parser.add_argument('--val_samples', type=int, default=100,
                        help='number of validation samples')
    parser.add_argument('--eval_samples', type=int, default=100,
                        help='number of validation samples')
    parser.add_argument('--opt', choices=('adam', 'sgd'), default='adam')

    parser.add_argument('--initial_lr', type=float, default=5e-4)
    parser.add_argument('--lr_decay_rate', type=float, default=0.5)
    parser.add_argument('--lr_decay_iter', type=int, default=50000)
    # openpoes/cifar: 5e-4, CPN: 1e-5
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    parser.add_argument('--val_iter', type=int, default=1000)
    parser.add_argument('--log_iter', type=int, default=20)
    parser.add_argument('--save_iter', type=int, default=2000)
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU')
    parser.add_argument('--coco_dir', help='path of COCO dataset directory')
    parser.add_argument('--out', '-o', default='result/test',
                        help='output directory')
    parser.add_argument('--resume', '-r', default='',
                        help='initialize the trainer from given file')

    # distillation params
    parser.add_argument('--distill', action='store_true')
    parser.add_argument('--only_soft', action='store_true',
                        help='train student model with only soft target')
    parser.add_argument('--comp_heat', action='store_true',
                        help='complete heatmap labels with output of teacher model')
    parser.add_argument('--comp_paf', action='store_true',
                        help='complete paf labels with output of teacher model')
    parser.add_argument('--modify', action='store_true',
                        help='modify output of teacher model for distillation' \
                        +'(not for label omplement)')
    parser.add_argument('--teacher_path', default=params['teacher_path'])
    parser.add_argument('--teacher_type', choices=params['teacher_types'],
                        default='single')

    parser.add_argument('--use_all_images', action='store_true')
    parser.add_argument('--use_ignore_mask', type=int, choices=(0, 1), default=1)
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print(json.dumps(vars(args), sort_keys=True, indent=4))

    np.random.seed(0)
    random.seed(0)

    # Prepare model
    if args.arch == 'posenet':
        model = params['archs'][args.arch](stages=args.stages)
    elif args.arch in ['resnet50-dilate', 'resnet101-dilate', 'resnet152-dilate']:
        model = params['archs'][args.arch](dilate=True)
    else:
        model = params['archs'][args.arch]()

    if args.arch == 'posenet':
        posenet.copy_vgg_params(model)
    elif args.arch in ['nn1']:
        nn1.copy_squeezenet_params(model.squeeze)
    elif args.arch in ['resnetfpn', 'pspnet', 'cpn']:
        chainer.serializers.load_npz('models/resnet50.npz', model.res)

    if args.initmodel:
        print('Load model from', args.initmodel)
        chainer.serializers.load_npz(args.initmodel, model)

    # Prepare teacher model for distillation
    if args.distill or args.comp_heat or args.comp_paf:
        teacher = posenet.PoseNet()
        serializers.load_npz(args.teacher_path, teacher)
        teacher.disable_update()
    else:
        teacher = None

    # Set up GPU
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
        if args.distill or args.comp_heat or args.comp_paf:
            teacher.to_gpu()

    # Set up an optimizer
    if args.opt == 'sgd':
        optimizer = optimizers.MomentumSGD(lr=args.initial_lr, momentum=0.9)
    elif args.opt == 'adam':
        optimizer = optimizers.Adam(alpha=1e-4, beta1=0.9, beta2=0.999, eps=1e-08)
    optimizer.setup(model)
    if args.opt == 'sgd':
        optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))

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
        elif args.arch in ['nn1']:
            model.squeeze.disable_update()
        if args.arch in ['resnet50', 'resnet101', 'resnet152',
                         'resnet50-dilate', 'resnet101-dilate', 'resnet152-dilate']:
            model.res.disable_update()

    # Load datasets
    coco_dir = args.coco_dir or params['coco_dir']
    coco_train = COCO(os.path.join(coco_dir, 'annotations/person_keypoints_train2017.json'))
    coco_val = COCO(os.path.join(coco_dir, 'annotations/person_keypoints_val2017.json'))
    train_loader = CocoDataLoader(coco_dir, coco_train, model.insize, mode='train',
                                  use_all_images=args.use_all_images,
                                  use_ignore_mask=args.use_ignore_mask,
                                  load_label=True)
    val_loader = CocoDataLoader(coco_dir, coco_val, model.insize, mode='val',
                                n_samples=args.val_samples,
                                use_ignore_mask=args.use_ignore_mask,
                                use_all_images=args.use_all_images,
                                load_label=True)
    # eval_loader = CocoDataLoader(coco_dir, coco_val, model, mode='eval', n_samples=args.eval_samples)

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

    val_interval = (10 if args.test else args.val_iter), 'iteration'
    log_interval = (1 if args.test else args.log_iter), 'iteration'

    trainer.extend(Validator(val_iter, model, teacher, device=args.gpu),
                   trigger=val_interval)
    # trainer.extend(Evaluator(coco_val, eval_iter, model, device=args.gpu),
    #                trigger=val_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr())
    trainer.extend(extensions.ExponentialShift(
        'lr' if args.opt == 'sgd' else 'alpha', args.lr_decay_rate),
        trigger=(args.lr_decay_iter, 'iteration'))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.ProgressBar(update_interval=1))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'main/loss', 'val/loss', 'main/paf', 'val/paf',
        'main/heat', 'val/heat' #'AP', 'AR'
    ]), trigger=log_interval)
    # trainer.extend(extensions.PlotReport(
    #     ['main/loss', 'val/loss'], x_key='iteration', file_name='loss.png'))

    if not args.test:
        trainer.extend(extensions.snapshot(), trigger=(args.save_iter, 'iteration'))
        trainer.extend(extensions.snapshot_object(
            model, 'model_iter_{.updater.iteration}'), trigger=val_interval)

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    # Save training parameters
    if not os.path.exists(args.out):
        os.makedirs(args.out)
    txt = '@{}'.format(datetime.datetime.now().strftime('%y%m%d_%H%M'))
    subprocess.call("touch '{}'".format(os.path.join(args.out, txt)), shell=True)
    with open(os.path.join(args.out, 'params.json'), 'w') as f:
        json.dump(vars(args), f)

    trainer.run()
