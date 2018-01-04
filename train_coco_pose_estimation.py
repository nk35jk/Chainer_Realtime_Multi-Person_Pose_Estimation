import os
import cv2
import copy
import json
import glob
import random
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

from entity import JointType, params, parse_args
from coco_data_loader import CocoDataLoader
from pose_detector import PoseDetector, draw_person_pose

from models import CocoPoseNet, posenet, nn1, resnetfpn, pspnet, student


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


def compute_loss(imgs, pafs_ys, heatmaps_ys, masks_ys, pafs_t, heatmaps_t,
                 pt_pafs, pt_heatmaps, ignore_mask, stuff_mask, compute_mask,
                 device):
    heatmap_loss_log = []
    paf_loss_log = []
    mask_loss_log = []
    total_loss = 0

    paf_masks = ignore_mask[:, None].repeat(pafs_t.shape[1], axis=1)
    heatmap_masks = ignore_mask[:, None].repeat(heatmaps_t.shape[1], axis=1)
    mask_masks = ignore_mask[:, None].repeat(2, axis=1)

    # compute loss on each stage
    for pafs_y, heatmaps_y, masks_y in zip(pafs_ys, heatmaps_ys, masks_ys):
        stage_pafs_t = pafs_t.copy()
        stage_heatmaps_t = heatmaps_t.copy()
        stage_mask_t = stuff_mask.copy().astype('f')
        stage_paf_masks = paf_masks.copy()
        stage_heatmap_masks = heatmap_masks.copy()
        stage_mask_masks = mask_masks.copy()

        if pafs_y.shape != stage_pafs_t.shape:
            stage_pafs_t = F.resize_images(stage_pafs_t, pafs_y.shape[2:]).data
            stage_heatmaps_t = F.resize_images(stage_heatmaps_t, pafs_y.shape[2:]).data
            stage_paf_masks = F.resize_images(stage_paf_masks.astype('f'), pafs_y.shape[2:]).data > 0
            stage_heatmap_masks = F.resize_images(stage_heatmap_masks.astype('f'), pafs_y.shape[2:]).data > 0
            stage_mask_masks = F.resize_images(stage_mask_masks.astype('f'), pafs_y.shape[2:]).data > 0

            if device >= 0:
                stage_mask_t = cuda.to_cpu(stage_mask_t)
            stage_mask_t = cv2.resize(stage_mask_t.transpose(1, 2, 0),
                pafs_y.shape[2:], interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)
            if device >= 0:
                stage_mask_t = cuda.to_gpu(stage_mask_t)

        xp = cuda.get_array_module(stage_mask_t)
        stage_mask_t = xp.stack((stage_mask_t, -(stage_mask_t-1)), axis=1)  # for mean_squared_error

        stage_pafs_t[stage_paf_masks == True] = pafs_y.data[stage_paf_masks == True]
        stage_heatmaps_t[stage_heatmap_masks == True] = heatmaps_y.data[stage_heatmap_masks == True]
        if compute_mask:
            stage_mask_t[stage_mask_masks == True] = masks_y.data[stage_mask_masks == True]
            stage_mask_t[(stage_mask_t == -1) | (stage_mask_t == 2)] = masks_y.data[(stage_mask_t == -1) | (stage_mask_t == 2)]

        pafs_loss = F.mean_squared_error(pafs_y, stage_pafs_t)
        heatmaps_loss = F.mean_squared_error(heatmaps_y, stage_heatmaps_t)
        # heatmaps_loss = F.sum(F.mean(F.squared_error(heatmaps_y, stage_heatmaps_t), axis=0))
        mask_loss = 0
        if compute_mask:
            # mask_loss = F.softmax_cross_entropy(masks_y, stage_mask_t)
            mask_loss = F.mean_squared_error(masks_y, stage_mask_t)

        if pt_pafs is None:
            total_loss += pafs_loss + heatmaps_loss + params['mask_loss_ratio'] * mask_loss
        else:
            soft_pafs_loss = F.mean_squared_error(pafs_y, pt_pafs)
            soft_heatmaps_loss = F.mean_squared_error(heatmaps_y, pt_heatmaps)
            total_loss += 0.5 * (pafs_loss + heatmaps_loss) + 0.5 * (soft_pafs_loss + soft_heatmaps_loss)

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
    elif args.arch in ['nn1', 'resnetfpn', 'psp', 'student']:
        x_data -= xp.array([104, 117, 123])
    x_data = x_data.transpose(0, 3, 1, 2)
    return x_data


class Updater(StandardUpdater):

    def __init__(self, iterator, model, teacher, optimizer, compute_mask, device=None):
        super(Updater, self).__init__(iterator, optimizer, device=device)
        self.teacher = teacher
        self.compute_mask = compute_mask

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
            elif args.arch in ['resnetfpn', 'pspnet']:
                optimizer.target.res.enable_update()
            elif args.arch in ['nn1', 'student']:
                model.squeeze.enable_update()

        batch = train_iter.next()

        imgs, pafs, heatmaps, ignore_mask, stuff_mask = self.converter(batch, self.device)

        x_data = preprocess(imgs)

        if self.compute_mask:
            pafs_ys, heatmaps_ys, masks_ys = optimizer.target(x_data)
        else:
            pafs_ys, heatmaps_ys = optimizer.target(x_data)
            masks_ys = [None] * len(pafs_ys)

        pt_pafs = pt_heatmaps = None
        if self.teacher:
            x_data = ((imgs.astype('f') / 255) - 0.5).transpose(0, 3, 1, 2)
            h1s, h2s = self.teacher(x_data)
            pt_pafs = h1s[-1].data
            pt_heatmaps = h2s[-1].data

        loss, paf_loss_log, heatmap_loss_log, mask_loss_log = compute_loss(
            imgs, pafs_ys, heatmaps_ys, masks_ys, pafs, heatmaps, pt_pafs,
            pt_heatmaps, ignore_mask, stuff_mask, self.compute_mask, self.device)

        reporter.report({
            'main/loss': loss,
            'main/paf': sum(paf_loss_log),
            'main/heat': sum(heatmap_loss_log),
            'main/mask': sum(mask_loss_log),
        })

        optimizer.target.cleargrads()
        loss.backward()
        optimizer.update()


class Validator(extensions.Evaluator):

    def __init__(self, iterator, model, teacher, compute_mask, device=None):
        super(Validator, self).__init__(iterator, model, device=device)
        self.iterator = iterator
        self.compute_mask = compute_mask
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
                imgs, pafs, heatmaps, ignore_mask, stuff_mask = self.converter(batch, self.device)
                with function.no_backprop_mode():
                    x_data = preprocess(imgs)

                    if self.compute_mask:
                        pafs_ys, heatmaps_ys, masks_ys = model(x_data)
                    else:
                        pafs_ys, heatmaps_ys = model(x_data)
                        masks_ys = [None] * len(pafs_ys)

                    pt_pafs = pt_heatmaps = None
                    if self.teacher:
                        x_data = ((imgs.astype('f') / 255) - 0.5).transpose(0, 3, 1, 2)
                        h1s, h2s = self.teacher(x_data)
                        pt_pafs = h1s[-1]
                        pt_heatmaps = h2s[-1]

                    loss, paf_loss_log, heatmap_loss_log, mask_loss_log = compute_loss(
                        imgs, pafs_ys, heatmaps_ys, masks_ys, pafs, heatmaps, pt_pafs,
                        pt_heatmaps, ignore_mask, stuff_mask, self.compute_mask, self.device)

                    observation['val/loss'] = cuda.to_cpu(loss.data)
                    observation['val/paf'] = sum(paf_loss_log)
                    observation['val/heat'] = sum(heatmap_loss_log)
                    observation['val/mask'] = sum(mask_loss_log)
            summary.add(observation)
        return summary.compute_mean()


class Evaluator(extensions.Evaluator):

    def __init__(self, cocoGt, iterator, model, compute_mask, device=None):
        super(Evaluator, self).__init__(iterator, model, device=device)
        self.cocoGt = cocoGt
        self.pose_detector = PoseDetector(model=model, device=device, precise=True, compute_mask=args.mask)

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
                poses = self.pose_detector(img)

                for pose in poses:
                    res_dict = {}
                    res_dict['category_id'] = 1
                    res_dict['image_id'] = img_id
                    res_dict['score'] = 1

                    keypoints = np.zeros((len(params['coco_joint_indices']), 3))
                    for joint, jt in zip(pose, JointType):
                        if joint is not None and jt in params['coco_joint_indices']:
                            i = params['coco_joint_indices'].index(jt)
                            keypoints[i] = joint
                    res_dict['keypoints'] = keypoints.ravel()
                    res.append(res_dict)

            img = draw_person_pose(img, poses)

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

    np.random.seed(0)
    random.seed(0)

    # Load the datasets
    coco_train = COCO(os.path.join(params['coco_dir'], 'annotations/person_keypoints_train2017.json'))
    coco_val = COCO(os.path.join(params['coco_dir'], 'annotations/person_keypoints_val2017.json'))
    train_loader = CocoDataLoader(coco_train, mode='train')
    val_loader = CocoDataLoader(coco_val, mode='val', n_samples=args.val_samples)
    eval_loader = CocoDataLoader(coco_val, mode='eval', n_samples=args.eval_samples)

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

    if not os.path.exists(args.out):
        os.makedirs(args.out)
    txt = '@{}'.format(datetime.datetime.now().strftime('%y%m%d_%H%M'))
    with open(os.path.join(args.out, txt), 'w') as f:
        pass
    with open(os.path.join(args.out, 'params.json'), 'w') as f:
        json.dump(vars(args), f)

    # Prepare model
    model = params['archs'][args.arch](compute_mask=args.mask)

    if args.arch == 'posenet':
        posenet.copy_vgg_params(model)
    elif args.arch in ['nn1', 'student']:
        nn1.copy_squeezenet_params(model.squeeze)
    elif args.arch in ['resnetfpn', 'pspnet']:
        chainer.serializers.load_npz('models/resnet50.npz', model.res)

    if args.initmodel:
        print('Load model from', args.initmodel)
        chainer.serializers.load_npz(args.initmodel, model)

    teacher = None
    if args.distill:
        teacher = posenet.PoseNet()
        serializers.load_npz('models/posenet_iter_200000', teacher)
        teacher.disable_update()
        if args.gpu >= 0:
            teacher.to_gpu()

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

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
        elif args.arch in ['resnetfpn', 'pspnet']:
            model.res.disable_update()
        elif args.arch in ['nn1', 'student']:
            model.squeeze.disable_update()

    # Set up a trainer
    updater = Updater(train_iter, model, teacher, optimizer, args.mask, device=args.gpu)
    trainer = training.Trainer(updater, (args.iteration, 'iteration'), args.out)

    val_interval = (10 if args.test else 1000), 'iteration'
    log_interval = (1 if args.test else 20), 'iteration'

    # trainer.extend(extensions.LinearShift('alpha', (1e-4, 5e-6), (1, 1000000)))
    trainer.extend(Validator(val_iter, model, teacher, args.mask, device=args.gpu),
                   trigger=val_interval)
    # trainer.extend(Evaluator(coco_val, eval_iter, model, args.mask, device=args.gpu),
    #                trigger=val_interval)
    # trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=val_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}'), trigger=val_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'main/loss', 'val/loss', 'main/paf', 'val/paf',
        'main/heat', 'val/heat',# 'main/mask', 'val/mask', 'AP', 'AR'
    ]), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=1))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()
