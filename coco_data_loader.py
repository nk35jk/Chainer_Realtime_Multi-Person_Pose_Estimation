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


class CocoDataLoader(DatasetMixin):

    def __init__(self, coco_dir, coco, insize, mode='train', n_samples=None,
                 use_all_images=False, use_ignore_mask=True, augment_data=True,
                 resize_data=True, load_label=False):
        self.coco_dir = coco_dir
        self.coco = coco
        assert mode in ['train', 'val', 'eval'], 'Data loading mode is invalid.'
        self.mode = mode
        self.catIds = coco.getCatIds(catNms=['person'])
        if use_all_images:
            self.imgIds = sorted(coco.getImgIds())
        else:
            self.imgIds = sorted(coco.getImgIds(catIds=self.catIds))
        if self.mode in ['val', 'eval'] and n_samples is not None:
            random.seed(2)
            self.imgIds = random.sample(self.imgIds, n_samples)
        print('{} images: {}'.format(mode, len(self)))
        self.insize = insize
        self.use_ignore_mask = bool(use_ignore_mask)
        self.augment_data_ = augment_data
        self.resize_data_ = resize_data
        self.load_label = load_label

    def __len__(self):
        return len(self.imgIds)

    def overlay_paf(self, img, paf, alpha=0.25, beta=0.75):
        hue = (np.arctan2(paf[1], paf[0]) / np.pi) / -2 + 0.5
        saturation = np.ones_like(hue)
        value = np.sqrt(paf[0] ** 2 + paf[1] ** 2) * 2  # twiced
        value[value > 1.0] = 1.0
        hsv_paf = np.stack([hue*180, saturation*255, value*255]).transpose(1, 2, 0)
        rgb_paf = cv2.cvtColor(hsv_paf.astype(np.uint8), cv2.COLOR_HSV2BGR)
        img = cv2.addWeighted(img, alpha, rgb_paf, beta, 0)
        return img

    def overlay_pafs(self, img, pafs, alpha=0.25, beta=0.75):
        mix_paf = np.zeros((2,) + img.shape[:-1])
        paf_flags = np.zeros(mix_paf.shape) # for constant paf

        for paf in pafs.reshape((int(pafs.shape[0]/2), 2,) + pafs.shape[1:]):
            paf_flags = paf != 0
            paf_flags += np.broadcast_to(paf_flags[0] | paf_flags[1], paf.shape)
            mix_paf += paf

        mix_paf[paf_flags > 0] /= paf_flags[paf_flags > 0]
        img = self.overlay_paf(img, mix_paf, alpha, beta)
        return img

    def overlay_heatmap(self, img, heatmap, alpha=0.25, beta=0.75):
        rgb_heatmap = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        img = cv2.addWeighted(img, alpha, rgb_heatmap, beta, 0)
        return img

    def overlay_ignore_mask(self, img, ignore_mask, alpha=0.3, beta=0.7):
        masked_img = img.copy()
        masked_img[ignore_mask] = [0, 0, 255]
        img = cv2.addWeighted(img, alpha, masked_img, beta, 0)
        # img = img * np.repeat((ignore_mask == 0).astype(np.uint8)[:, :, None], 3, axis=2)
        return img

    def get_pose_bboxes(self, poses):
        pose_bboxes = []
        for pose in poses:
            x1 = pose[pose[:, 2] > 0][:, 0].min()
            y1 = pose[pose[:, 2] > 0][:, 1].min()
            x2 = pose[pose[:, 2] > 0][:, 0].max()
            y2 = pose[pose[:, 2] > 0][:, 1].max()
            pose_bboxes.append([x1, y1, x2, y2])
        pose_bboxes = np.array(pose_bboxes)
        return pose_bboxes

    def resize_data(self, img, ignore_mask, poses, shape, label=None):
        """resize img, mask and annotations"""
        img_h, img_w, _ = img.shape

        resized_img = cv2.resize(img, shape)
        ignore_mask = cv2.resize(ignore_mask.astype(np.uint8), shape).astype('bool')
        poses[:, :, :2] = (poses[:, :, :2] * np.array(shape) / np.array((img_w, img_h)))

        if label is not None:
            label = cv2.resize(label.transpose(1, 2, 0), shape).transpose(2, 0, 1)
        return resized_img, ignore_mask, poses, label

    def random_resize_img(self, img, ignore_mask, poses, label=None):
        h, w, _ = img.shape

        if len(poses) > 0:
            joint_bboxes = self.get_pose_bboxes(poses)
            bbox_sizes = ((joint_bboxes[:, 2:] - joint_bboxes[:, :2] + 1)**2).sum(axis=1)**0.5

            min_scale = params['min_box_size']/bbox_sizes.min()
            max_scale = params['max_box_size']/bbox_sizes.max()
            # print(len(bbox_sizes))
            # print('min: {}, max: {}'.format(min_scale, max_scale))

            min_scale = min(max(min_scale, params['min_scale']), 1)
            max_scale = min(max(max_scale, 1), params['max_scale'])
            # print('min: {:.2f}, max: {:.2f}'.format(min_scale, max_scale))

            scale = float((max_scale - min_scale) * random.random() + min_scale)
            # print('scale: {:.2f}'.format(scale))

        else:
            scale = params['max_scale'] + random.random() * (params['max_scale'] - params['min_scale'])

        shape = (round(w * scale), round(h * scale))

        resized_img, resized_mask, resized_poses, label = self.resize_data(img, ignore_mask, poses, shape, label)
        return resized_img, resized_mask, poses, label

    def random_rotate_img(self, img, mask, poses, label=None):
        h, w, _ = img.shape
        # degree = (random.random() - 0.5) * 2 * params['max_rotate_degree']
        degree = np.random.randn() * 10
        degree = max(degree, -params['max_rotate_degree'])
        degree = min(degree, params['max_rotate_degree'])
        rad = degree * math.pi / 180
        center = (w / 2, h / 2)
        R = cv2.getRotationMatrix2D(center, degree, 1)
        bbox = (w*abs(math.cos(rad)) + h*abs(math.sin(rad)), w*abs(math.sin(rad)) + h*abs(math.cos(rad)))
        R[0, 2] += bbox[0] / 2 - center[0]
        R[1, 2] += bbox[1] / 2 - center[1]
        rotate_img = cv2.warpAffine(img, R, (int(bbox[0]+0.5), int(bbox[1]+0.5)), flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=(104, 117, 123))
        rotate_mask = cv2.warpAffine(mask.astype('uint8')*255, R, (int(bbox[0]+0.5), int(bbox[1]+0.5))) > 0
        if label is not None:
            label = cv2.warpAffine(label.transpose(1, 2, 0), R, (int(bbox[0]+0.5), int(bbox[1]+0.5)), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0).transpose(2, 0, 1) # last channel

        tmp_poses = np.ones_like(poses)
        tmp_poses[:, :, :2] = poses[:, :, :2].copy()
        tmp_rotate_poses = np.dot(tmp_poses, R.T)  # apply rotation matrix to the poses
        rotate_poses = poses.copy()  # to keep visibility flag
        rotate_poses[:, :, :2] = tmp_rotate_poses
        return rotate_img, rotate_mask, rotate_poses, label

    def random_crop_img(self, img, ignore_mask, poses, label=None):
        h, w, _ = img.shape

        if len(poses) > 0:
            joint_bboxes = self.get_pose_bboxes(poses)
            bbox = random.choice(joint_bboxes)  # select a bbox randomly
            bbox_center = bbox[:2] + (bbox[2:] - bbox[:2])/2

            r_xy = np.random.rand(2)
            perturb = ((r_xy - 0.5) * 2 * params['center_perterb_max'])
            center = (bbox_center + perturb + 0.5).astype('i')

            crop_img = np.zeros((self.insize, self.insize, 3), 'uint8') + (104, 117, 123)
            crop_mask = np.zeros((self.insize, self.insize), 'bool')

            offset = (center - (self.insize-1)/2 + 0.5).astype('i')
            offset_ = (center + (self.insize-1)/2 - (w-1, h-1) + 0.5).astype('i')

            x1, y1 = (center - (self.insize-1)/2 + 0.5).astype('i')
            x2, y2 = (center + (self.insize-1)/2 + 0.5).astype('i')

            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(x2, w-1)
            y2 = min(y2, h-1)

            x_from = -offset[0] if offset[0] < 0 else 0
            y_from = -offset[1] if offset[1] < 0 else 0
            x_to = self.insize - offset_[0] - 1 if offset_[0] >= 0 else self.insize - 1
            y_to = self.insize - offset_[1] - 1 if offset_[1] >= 0 else self.insize - 1

            crop_img[y_from:y_to+1, x_from:x_to+1] = img[y1:y2+1, x1:x2+1].copy()
            crop_mask[y_from:y_to+1, x_from:x_to+1] = ignore_mask[y1:y2+1, x1:x2+1].copy()

            if label is not None:
                crop_label = np.zeros((57, self.insize, self.insize), 'float32')
                crop_label[:, y_from:y_to+1, x_from:x_to+1] = label[:, y1:y2+1, x1:x2+1].copy()
                label = crop_label

            poses[:, :, :2] -= offset
        else:
            crop_img = np.zeros((self.insize, self.insize, 3), 'uint8') + (104, 117, 123)
            crop_mask = np.zeros((self.insize, self.insize), 'bool')

            if h >= self.insize and w >= self.insize:
                y1 = random.randint(0, h - self.insize)
                x1 = random.randint(0, w - self.insize)
                crop_img = img[y1:y1+self.insize, x1:x1+self.insize].copy()
                if label is not None:
                    crop_label = label[:, y1:y1+self.insize, x1:x1+self.insize].copy()
                    label = crop_label
            else:
                center = np.array([w/2, h/2]).astype('i')

                offset = (center - (self.insize-1)/2 + 0.5).astype('i')
                offset_ = (center + (self.insize-1)/2 - (w-1, h-1) + 0.5).astype('i')

                x1, y1 = (center - (self.insize-1)/2 + 0.5).astype('i')
                x2, y2 = (center + (self.insize-1)/2 + 0.5).astype('i')

                x1 = max(x1, 0)
                y1 = max(y1, 0)
                x2 = min(x2, w-1)
                y2 = min(y2, h-1)

                x_from = -offset[0] if offset[0] < 0 else 0
                y_from = -offset[1] if offset[1] < 0 else 0
                x_to = self.insize - offset_[0] - 1 if offset_[0] >= 0 else self.insize - 1
                y_to = self.insize - offset_[1] - 1 if offset_[1] >= 0 else self.insize - 1

                crop_img[y_from:y_to+1, x_from:x_to+1] = img[y1:y2+1, x1:x2+1].copy()

                if label is not None:
                    crop_label = np.zeros((57, self.insize, self.insize), 'float32')
                    crop_label[:, y_from:y_to+1, x_from:x_to+1] = label[:, y1:y2+1, x1:x2+1].copy()
                    label = crop_label

        return crop_img.astype('uint8'), crop_mask, poses, label

    def distort_color(self, img):
        img_max = np.broadcast_to(np.array(255, dtype=np.uint8), img.shape[:-1])
        img_min = np.zeros(img.shape[:-1], dtype=np.uint8)

        hsv_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV).astype(np.int32)
        hsv_img[:, :, 0] = np.maximum(np.minimum(hsv_img[:, :, 0] - 10 + np.random.randint(20 + 1), img_max), img_min) # hue
        hsv_img[:, :, 1] = np.maximum(np.minimum(hsv_img[:, :, 1] - 40 + np.random.randint(80 + 1), img_max), img_min) # saturation
        hsv_img[:, :, 2] = np.maximum(np.minimum(hsv_img[:, :, 2] - 30 + np.random.randint(60 + 1), img_max), img_min) # value
        hsv_img = hsv_img.astype(np.uint8)

        distorted_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
        return distorted_img

    def flip_img(self, img, mask, poses, label=None):
        flipped_img = cv2.flip(img, 1)
        flipped_mask = cv2.flip(mask.astype(np.uint8), 1).astype('bool')
        poses[:, :, 0] = img.shape[1] - 1 - poses[:, :, 0]

        def swap_joints(poses, joint_type_1, joint_type_2):
            tmp = poses[:, joint_type_1].copy()
            poses[:, joint_type_1] = poses[:, joint_type_2]
            poses[:, joint_type_2] = tmp

        swap_joints(poses, JointType.LeftEye, JointType.RightEye)
        swap_joints(poses, JointType.LeftEar, JointType.RightEar)
        swap_joints(poses, JointType.LeftShoulder, JointType.RightShoulder)
        swap_joints(poses, JointType.LeftElbow, JointType.RightElbow)
        swap_joints(poses, JointType.LeftHand, JointType.RightHand)
        swap_joints(poses, JointType.LeftWaist, JointType.RightWaist)
        swap_joints(poses, JointType.LeftKnee, JointType.RightKnee)
        swap_joints(poses, JointType.LeftFoot, JointType.RightFoot)

        def swap_heatmaps(label, joint_type_1, joint_type_2):
            joint_type_1 += 38
            joint_type_2 += 38

            tmp = label[joint_type_1].copy()
            label[joint_type_1] = label[joint_type_2]
            label[joint_type_2] = tmp

        def swap_pafs(label, limb1, limb2):
            tmp = label[limb1*2:limb1*2+2].copy()
            label[limb1*2:limb1*2+2] = label[limb2*2:limb2*2+2]
            label[limb2*2:limb2*2+2] = tmp

        if label is not None:
            swap_heatmaps(label, JointType.LeftEye, JointType.RightEye)
            swap_heatmaps(label, JointType.LeftEar, JointType.RightEar)
            swap_heatmaps(label, JointType.LeftShoulder, JointType.RightShoulder)
            swap_heatmaps(label, JointType.LeftElbow, JointType.RightElbow)
            swap_heatmaps(label, JointType.LeftHand, JointType.RightHand)
            swap_heatmaps(label, JointType.LeftWaist, JointType.RightWaist)
            swap_heatmaps(label, JointType.LeftKnee, JointType.RightKnee)
            swap_heatmaps(label, JointType.LeftFoot, JointType.RightFoot)
            swap_pafs(label, 0, 3)
            swap_pafs(label, 1, 4)
            swap_pafs(label, 2, 5)
            swap_pafs(label, 6, 10)
            swap_pafs(label, 7, 11)
            swap_pafs(label, 8, 12)
            swap_pafs(label, 9, 13)
            swap_pafs(label, 15, 16)
            swap_pafs(label, 17, 18)
        return flipped_img, flipped_mask, poses, label

    def augment_data(self, img, ignore_mask, poses, label=None):
        aug_img = img.copy()
        aug_img, ignore_mask, poses, label = self.random_resize_img(aug_img, ignore_mask, poses, label)
        aug_img, ignore_mask, poses, label = self.random_rotate_img(aug_img, ignore_mask, poses, label)
        aug_img, ignore_mask, poses, label = self.random_crop_img(aug_img, ignore_mask, poses, label)
        if np.random.randint(2):
            aug_img = self.distort_color(aug_img)
        if np.random.randint(2):
            aug_img, ignore_mask, poses, label = self.flip_img(aug_img, ignore_mask, poses, label)

        return aug_img, ignore_mask, poses, label

    def gen_heatmap(self, shape, joint, sigma):
        """return shape: (height, width)"""
        x, y, v = joint
        if v == 0:
            return np.zeros(shape)
        grid_x = np.tile(np.arange(shape[1]), (shape[0], 1))
        grid_y = np.tile(np.arange(shape[0]), (shape[1], 1)).transpose()
        grid_distance = (grid_x - x) ** 2 + (grid_y - y) ** 2
        gaussian_heatmap = np.exp(-0.5 * grid_distance / sigma**2)
        return gaussian_heatmap

    def gen_heatmaps(self, img, poses, scales, sigma):
        heatmaps = np.zeros((0,) + img.shape[:-1])
        heatmap_max = np.zeros(img.shape[:-1])
        for joint_index in range(len(JointType)):
            heatmap = np.zeros(img.shape[:-1])
            for pose, scale in zip(poses, scales):
                if pose[joint_index, 2] > 0:
                    jointmap = self.gen_heatmap(img.shape[:-1], pose[joint_index], sigma*scale)
                    heatmap[jointmap > heatmap] = jointmap[jointmap > heatmap]
                    heatmap_max[jointmap > heatmap_max] = jointmap[jointmap > heatmap_max]
            heatmaps = np.vstack((heatmaps, heatmap.reshape((1,) + heatmap.shape)))
        bg_heatmap = 1 - heatmap_max # background channel
        heatmaps = np.vstack((heatmaps, bg_heatmap[None]))
        return heatmaps.astype('f')

    def gen_heatmaps2(self, img, poses, scales, sigma):
        heatmaps = np.zeros((len(JointType),) + img.shape[:-1])

        if len(poses) > 0:
            jointmaps = []
            for pose, scale in zip(poses, scales):
                jointmaps.append(np.stack([self.gen_heatmap(img.shape[:-1], pose[i], sigma*scale) for i in range(len(JointType))]))
            heatmaps = np.array(jointmaps).max(axis=0)

        bg_heatmap = 1 - heatmaps.max(axis=0) # background channel
        heatmaps = np.vstack((heatmaps, bg_heatmap[None]))
        return heatmaps.astype('f')

    def gen_constant_paf(self, shape, joint_from, joint_to, paf_sigma, scale=1):
        """return shape: (2, height, width)"""
        if np.array_equal(joint_from, joint_to): # same joint
            return np.zeros((2,) + shape)

        joint_distance = np.linalg.norm(joint_to - joint_from)
        unit_vector = (joint_to - joint_from) / joint_distance
        rad = np.pi / 2
        rot_matrix = np.array([[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]])
        unit_vector_v = np.dot(rot_matrix, unit_vector)
        grid_x = np.tile(np.arange(shape[1]), (shape[0], 1))
        grid_y = np.tile(np.arange(shape[0]), (shape[1], 1)).transpose()
        inner_product_h = unit_vector[0] * (grid_x - joint_from[0]) + unit_vector[1] * (grid_y - joint_from[1])
        paf_flag_h = (0 <= inner_product_h) & (inner_product_h <= joint_distance)
        inner_product_v = unit_vector_v[0] * (grid_x - joint_from[0]) + unit_vector_v[1] * (grid_y - joint_from[1])
        paf_flag_v = np.abs(inner_product_v) <= paf_sigma
        paf_flag = paf_flag_h & paf_flag_v
        constant_paf = np.stack((paf_flag, paf_flag)) * np.broadcast_to(unit_vector, shape + (2,)).transpose(2, 0, 1)
        return constant_paf

    def gen_round_constant_paf(self, shape, joint_from, joint_to, sigma):
        """return shape: (2, height, width)"""
        v_from = joint_from[-1]
        v_to = joint_to[-1]
        joint_from = joint_from[:-1]
        joint_to = joint_to[:-1]

        if np.array_equal(joint_from, joint_to) or v_from == 0 or v_to == 0: # same joint
            return np.zeros((2,) + shape)

        joint_distance = np.linalg.norm(joint_to - joint_from)
        unit_vector = (joint_to - joint_from) / joint_distance
        rad = np.pi / 2
        rot_matrix = np.array([[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]])
        unit_vector_v = np.dot(rot_matrix, unit_vector)
        grid_x = np.tile(np.arange(shape[1]), (shape[0], 1)).astype('i')
        grid_y = np.tile(np.arange(shape[0]), (shape[1], 1)).transpose().astype('i')
        inner_product_h = unit_vector[0] * (grid_x - joint_from[0]) + unit_vector[1] * (grid_y - joint_from[1]) # 730
        paf_flag_h = (0 <= inner_product_h) & (inner_product_h <= joint_distance) # 80
        close_to_joint_from = ((joint_from[0] - grid_x)**2 + (joint_from[1] - grid_y)**2)**0.5 < sigma # 3300
        close_to_joint_to = ((joint_to[0] - grid_x)**2 + (joint_to[1] - grid_y)**2)**0.5 < sigma # 3300
        inner_product_v = unit_vector_v[0] * (grid_x - joint_from[0]) + unit_vector_v[1] * (grid_y - joint_from[1]) # 820
        paf_flag_v = np.abs(inner_product_v) <= sigma # 90
        paf_flag = paf_flag_h & paf_flag_v | close_to_joint_from | close_to_joint_to # 33
        constant_paf = np.stack((paf_flag, paf_flag)) * np.broadcast_to(unit_vector, shape + (2,)).transpose(2, 0, 1) # 220
        return constant_paf

    def gen_gaussian_paf(self, shape, joint_from, joint_to, sigma):
        v_from = joint_from[-1]
        v_to = joint_to[-1]
        joint_from = joint_from[:-1]
        joint_to = joint_to[:-1]

        if np.array_equal(joint_from, joint_to):
            return np.zeros((2,) + shape)

        joint_distance = np.linalg.norm(joint_to - joint_from)
        unit_vector = (joint_to - joint_from) / joint_distance
        rad = np.pi / 2
        rot_matrix = np.array([[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]])
        unit_vector_v = np.dot(rot_matrix, unit_vector)
        grid_x = np.tile(np.arange(shape[1]), (shape[0], 1))
        grid_y = np.tile(np.arange(shape[0]), (shape[1], 1)).transpose()
        inner_product_h = unit_vector[0] * (grid_x - joint_from[0]) + unit_vector[1] * (grid_y - joint_from[1])
        paf_flag_h = (0 <= inner_product_h) & (inner_product_h <= joint_distance)
        inner_product_v = unit_vector_v[0] * (grid_x - joint_from[0]) + unit_vector_v[1] * (grid_y - joint_from[1])
        gauss = np.exp(-np.abs(inner_product_v) ** 2 / sigma ** 2)

        grid_distance1 = (grid_x - joint_from[0]) ** 2 + (grid_y - joint_from[1]) ** 2
        grid_distance2 = (grid_x - joint_to[0]) ** 2 + (grid_y - joint_to[1]) ** 2
        gaussian_heatmap = np.exp(-0.5 * grid_distance1 / sigma**2)

        gaussian_paf = np.stack((paf_flag_h * gauss, paf_flag_h * gauss)) * np.broadcast_to(unit_vector, shape + (2,)).transpose(2, 0, 1)
        return gaussian_paf

    def gen_pafs(self, img, poses, scales, sigma):
        pafs = np.zeros((0,) + img.shape[:-1])

        for limb in params['limbs_point']:
            paf = np.zeros((2,) + img.shape[:-1])
            paf_flags = np.zeros(paf.shape)

            for pose, scale in zip(poses, scales):
                joint_from, joint_to = pose[limb]
                if joint_from[2] > 0 and joint_to[2] > 0:
                    limb_paf = self.gen_round_constant_paf(img.shape[:-1], joint_from, joint_to, sigma*scale)
                    limb_paf_flags = limb_paf != 0
                    paf_flags += np.broadcast_to(limb_paf_flags[0] | limb_paf_flags[1], limb_paf.shape)
                    paf += limb_paf

            paf[paf_flags > 0] /= paf_flags[paf_flags > 0]
            pafs = np.vstack((pafs, paf))
        return pafs.astype('f')

    def gen_pafs2(self, img, poses, scales, sigma):
        pafs = np.zeros((len(params['limbs_point'])*2,) + img.shape[:-1])
        pafs_flag = np.zeros((len(params['limbs_point'])*2,) + img.shape[:-1]).astype('i')
        for pose, scale in zip(poses, scales):
            joint_froms, joint_tos = [], []
            for limb in params['limbs_point']:
                joint_pair = pose[limb]
                joint_froms.append(joint_pair[0])
                joint_tos.append(joint_pair[1])

            tmp_pafs = np.concatenate([self.gen_round_constant_paf(img.shape[:-1],from_, to, sigma*scale)
                                       for from_, to in zip(joint_froms, joint_tos)])
            pafs += tmp_pafs
            pafs_flag += tmp_pafs != 0

        pafs[pafs_flag > 0] /= pafs_flag[pafs_flag > 0]
        return pafs.astype('f')

    def get_img_annotation(self, ind=None, img_id=None):
        """インデックスまたは img_id から coco annotation dataを抽出、条件に満たない場合は空リストを返す """

        if ind is not None:
            img_id = self.imgIds[ind]
        anno_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)

        person_cnt = 0
        valid_annotations = []
        if len(anno_ids) > 0:
            annotations = self.coco.loadAnns(anno_ids)

            for ann in annotations:
                # if too few keypoints or too small
                if ann['category_id'] == 1 and ann['num_keypoints'] >= params['min_keypoints'] and ann['area'] > params['min_area']:
                    person_cnt += 1
                    valid_annotations.append(ann)

                # print('---')
                # print(ann['bbox'])
                # print(ann['num_keypoints'])
                # print(ann['area'])
                # print('---')

        if self.mode == 'train':
            img_path = os.path.join(self.coco_dir, 'train2017', '{:012d}.jpg'.format(img_id))
            mask_path = os.path.join(self.coco_dir, 'ignore_mask_train2017', '{:012d}.png'.format(img_id))
            label_path = os.path.join(self.coco_dir, 'labels_train2017', '{:012d}.npy'.format(img_id))
        else:
            img_path = os.path.join(self.coco_dir, 'val2017', '{:012d}.jpg'.format(img_id))
            mask_path = os.path.join(self.coco_dir, 'ignore_mask_val2017', '{:012d}.png'.format(img_id))
            label_path = os.path.join(self.coco_dir, 'labels_val2017', '{:012d}.npy'.format(img_id))

        img = cv2.imread(img_path)
        ignore_mask = cv2.imread(mask_path, 0)
        if ignore_mask is None or self.use_ignore_mask == False:
            ignore_mask = np.zeros(img.shape[:2], 'bool')
        else:
            ignore_mask = ignore_mask == 255

        label = None
        if self.load_label and os.path.exists(label_path):
            label = np.load(label_path)

        if self.mode == 'eval':
            return img, img_id, annotations, ignore_mask, label
        return img, img_id, valid_annotations, ignore_mask, label

    def parse_coco_annotation(self, annotations):
        """coco annotation dataのアノテーションをposes配列に変換"""
        poses = np.zeros((0, len(JointType), 3), dtype=np.int32)

        if len(annotations) == 0:
            return poses

        for ann in annotations:
            ann_pose = np.array(ann['keypoints']).reshape(-1, 3)
            pose = np.zeros((1, len(JointType), 3), dtype=np.int32)

            # convert poses position
            for i, joint_index in enumerate(params['coco_joint_indices']):
                pose[0][joint_index] = ann_pose[i]

            # compute neck position
            if pose[0][JointType.LeftShoulder][2] > 0 and pose[0][JointType.RightShoulder][2] > 0:
                pose[0][JointType.Neck][0] = int((pose[0][JointType.LeftShoulder][0] + pose[0][JointType.RightShoulder][0]) / 2)
                pose[0][JointType.Neck][1] = int((pose[0][JointType.LeftShoulder][1] + pose[0][JointType.RightShoulder][1]) / 2)
                pose[0][JointType.Neck][2] = 2

            poses = np.vstack((poses, pose))

        gt_pose = np.array(ann['keypoints']).reshape(-1, 3)
        return poses

    def gen_labels(self, img, annotations, ignore_mask, label=None):
        poses = self.parse_coco_annotation(annotations)
        scales = np.array([i['area']/params['area_basis'] for i in annotations])**0.5
        scales = scales.clip(max=params['max_ratio'])

        if self.augment_data_:
            img, ignore_mask, poses, label = self.augment_data(img, ignore_mask, poses, label)
        if self.resize_data_:
            img, ignore_mask, poses, label = self.resize_data(img, ignore_mask, poses, (self.insize, self.insize), label)

        if label is None:
            heatmaps = self.gen_heatmaps(img, poses, scales, params['heatmap_sigma'])
            pafs = self.gen_pafs(img, poses, scales, params['paf_sigma'])
            ignore_mask = cv2.morphologyEx(ignore_mask.astype('uint8'), cv2.MORPH_DILATE, np.ones((16, 16))).astype('bool')
        else:
            pafs, heatmaps = np.split(label, [len(params['limbs_point'])*2])

        return img, pafs, heatmaps, ignore_mask

    def get_example(self, i, img_id=None):
        self.img_id = img_id

        if img_id:
            img, img_id, annotations, ignore_mask, label = self.get_img_annotation(img_id=img_id)
        else:
            img, img_id, annotations, ignore_mask, label = self.get_img_annotation(ind=i)

        if self.mode == 'eval':
            return img, annotations, img_id

        resized_img, pafs, heatmaps, ignore_mask = self.gen_labels(img, annotations, ignore_mask, label)
        return resized_img, pafs, heatmaps, ignore_mask


if __name__ == '__main__':
    mode = 'val'  # train, val
    coco = COCO(os.path.join(params['coco_dir'], 'annotations/person_keypoints_{}2017.json'.format(mode)))
    data_loader = CocoDataLoader(params['coco_dir'], coco, params['insize'],
                                 mode=mode, use_all_images=False, use_ignore_mask=True,
                                 augment_data=True, resize_data=True, load_label=True)

    # cv2.namedWindow('w', cv2.WINDOW_NORMAL)

    sum_paf_avg_norm = 0
    sum_heatmap_avg_norm = 0

    for i in range(len(data_loader)):
        img, img_id, annotations, ignore_mask, label = data_loader.get_img_annotation(ind=i)

        print('img_id: {}'.format(img_id))

        if len(annotations) == 0:
            continue

        resized_img, pafs, heatmaps, ignore_mask = data_loader.gen_labels(img, annotations, ignore_mask)

        # resize to view
        shape = resized_img.shape[1::-1]
        pafs = cv2.resize(pafs.transpose(1, 2, 0), shape).transpose(2, 0, 1)
        heatmaps = cv2.resize(heatmaps.transpose(1, 2, 0), shape).transpose(2, 0, 1)
        ignore_mask = cv2.resize(ignore_mask.astype(np.uint8)*255, shape) > 0

        paf_avg_norm = np.linalg.norm(pafs, axis=0).mean()
        heatmap_avg_norm = np.linalg.norm(heatmaps[:-1], axis=0).mean()

        sum_paf_avg_norm += paf_avg_norm
        sum_heatmap_avg_norm += heatmap_avg_norm
        # print('{:.3f}'.format(sum_paf_avg_norm))
        # print('{:.3f}'.format(sum_heatmap_avg_norm))

        # overlay labels
        img_to_show = resized_img.copy()
        img_to_show = data_loader.overlay_pafs(img_to_show, pafs, .2, .8)
        # img_to_show = data_loader.overlay_heatmap(img_to_show, heatmaps[:-1].max(axis=0), .5, .5)
        # img_to_show = data_loader.overlay_ignore_mask(img_to_show, ignore_mask, .5, .5)

        cv2.imshow('w', np.hstack([resized_img, img_to_show]))
        # cv2.imwrite('result/label_ex/{:08d}_gt.jpg'.format(img_ids[i]), np.hstack([resized_img, img_to_show]))
        k = cv2.waitKey(0)
        if k == ord('q'):
            sys.exit()
        elif k == ord('d'):
            import ipdb; ipdb.set_trace()
