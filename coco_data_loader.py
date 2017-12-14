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
    def __init__(self, coco, mode='train', n_samples=None):
        self.coco = coco
        assert mode in ['train', 'val', 'eval'], 'Data loading mode is invalid.'
        self.mode = mode
        self.catIds = coco.getCatIds(catNms=['person'])
        self.imgIds = sorted(coco.getImgIds(catIds=self.catIds))
        if self.mode in ['val', 'eval'] and n_samples is not None:
            random.seed(2)
            self.imgIds = random.sample(self.imgIds, n_samples)
        print('{} images: {}'.format(mode, len(self)))

    def __len__(self):
        return len(self.imgIds)

    def overlay_paf(self, img, paf):
        hue = ((np.arctan2(paf[1], paf[0]) / np.pi) / -2 + 0.5)
        saturation = np.sqrt(paf[0] ** 2 + paf[1] ** 2)
        saturation[saturation > 1.0] = 1.0
        value = saturation.copy()
        hsv_paf = np.vstack((hue[np.newaxis], saturation[np.newaxis], value[np.newaxis])).transpose(1, 2, 0)
        rgb_paf = cv2.cvtColor((hsv_paf * 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        img = cv2.addWeighted(img, 0.7, rgb_paf, 0.3, 0)
        return img

    def overlay_pafs(self, img, pafs):
        mix_paf = np.zeros((2,) + img.shape[:-1])
        paf_flags = np.zeros(mix_paf.shape) # for constant paf

        for paf in pafs.reshape((int(pafs.shape[0]/2), 2,) + pafs.shape[1:]):
            paf_flags = paf != 0
            paf_flags += np.broadcast_to(paf_flags[0] | paf_flags[1], paf.shape)
            mix_paf += paf

        mix_paf[paf_flags > 0] /= paf_flags[paf_flags > 0]
        img = self.overlay_paf(img, mix_paf)
        return img

    def overlay_heatmap(self, img, heatmap):
        rgb_heatmap = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        img = cv2.addWeighted(img, 0.7, rgb_heatmap, 0.3, 0)
        return img

    def overlay_ignore_mask(self, img, ignore_mask):
        img = img * np.repeat((ignore_mask == 0).astype(np.uint8)[:, :, None], 3, axis=2)
        return img

    def overlay_stuff_mask(self, img, stuff_mask, n_class):
        hue = stuff_mask / (n_class - 1)
        saturation = np.ones_like(hue)
        value = saturation.copy()
        value[stuff_mask == -1] = 0
        hsv_stuff_mask = np.vstack((hue[None], saturation[None], value[None])).transpose(1, 2, 0)
        rgb_hsv_stuff_mask = cv2.cvtColor((hsv_stuff_mask * 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        img = cv2.addWeighted(img, 0.3, rgb_hsv_stuff_mask, 0.7, 0)
        return img

    def random_resize_img(self, img, ignore_mask, stuff_mask, joints):
        h, w, _ = img.shape
        joint_bboxes = self.get_joint_bboxes(joints)
        bbox_sizes = ((joint_bboxes[:, 2:] - joint_bboxes[:, :2] + 1)**2).sum(axis=1)**0.5
        print(len(bbox_sizes))

        min_scale = (params['target_dist']*params['insize'])/bbox_sizes.min()
        max_scale = (params['target_dist']*params['insize'])/bbox_sizes.max()

        print('min: {}, max: {}'.format(min_scale, max_scale))

        min_scale = min(min_scale, 1)
        max_scale = max(max_scale, 1)

        r = random.random()
        scale = float((max_scale - min_scale) * r + min_scale)
        shape = (round(w * scale), round(h * scale))
        # print(img.shape)
        # print(shape)

        resized_img, resized_mask, resized_joints, resized_stuff = self.resize_data(img, ignore_mask, joints, stuff_mask, shape)
        return resized_img, resized_mask, resized_stuff, joints

    def random_rotate_img(self, img, mask, stuff_mask, joints, max_rotate_degree):
        h, w, _ = img.shape
        r = random.random()
        degree = (r - 0.5) * 2 * max_rotate_degree
        rad = degree * math.pi / 180
        center = (w / 2, h / 2)
        R = cv2.getRotationMatrix2D(center, degree, 1)
        bbox = (w*abs(math.cos(rad)) + h*abs(math.sin(rad)), w*abs(math.sin(rad)) + h*abs(math.cos(rad)))
        R[0, 2] += bbox[0] / 2 - center[0]
        R[1, 2] += bbox[1] / 2 - center[1]
        rotate_img = cv2.warpAffine(img, R, (round(bbox[0]), round(bbox[1])), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(104, 117, 123))
        rotate_mask = cv2.warpAffine(mask.astype('uint8')*255, R, (round(bbox[0]), round(bbox[1]))) > 0
        rotate_stuff_mask = cv2.warpAffine(stuff_mask, R, (round(bbox[0]), round(bbox[1])), flags=cv2.INTER_NEAREST)

        tmp_joints = np.ones_like(joints)
        tmp_joints[:, :, :2] = joints[:, :, :2].copy()
        tmp_rotate_joints = np.dot(tmp_joints, R.T)  # apply rotation matrix to the joints
        rotate_joints = joints.copy()  # to keep visibility flag
        rotate_joints[:, :, :2] = tmp_rotate_joints
        return rotate_img, rotate_mask, rotate_stuff_mask, rotate_joints

    def random_crop_img(self, img, ignore_mask, stuff_mask, joints):
        h, w, _ = img.shape
        insize = params['insize']
        joint_bboxes = self.get_joint_bboxes(joints)
        bbox = random.choice(joint_bboxes)  # select a bbox randomly
        bbox_center = (bbox[:2] + (bbox[2:] - bbox[:2])/2).round().astype('i')

        r_xy = np.random.rand(2)
        perturb = ((r_xy - 0.5) * 2 * params['center_perterb_max']).round().astype('i')
        center = bbox_center + perturb

        crop_img = np.zeros((insize, insize, 3), 'uint8') + (104, 117, 123)
        crop_mask = np.zeros((insize, insize), 'bool')
        crop_stuff = np.zeros((insize, insize), 'int32')

        offset = (center - (insize/2)).round().astype('i')
        offset_ = (center + (insize/2)).round().astype('i') - 1 - (w-1, h-1)
        x1, y1 = np.round(center - insize/2).astype('i')
        x2, y2 = np.round(center + insize/2).astype('i') - 1
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, w)
        y2 = min(y2, h)

        x_from = -offset[0] if offset[0] < 0 else 0
        y_from = -offset[1] if offset[1] < 0 else 0
        x_to = insize - offset_[0] - 1 if offset_[0] > 0 else insize - 1
        y_to = insize - offset_[1] - 1 if offset_[1] > 0 else insize - 1

        crop_img[y_from:y_to+1, x_from:x_to+1] = img[y1:y2+1, x1:x2+1].copy()
        crop_mask[y_from:y_to+1, x_from:x_to+1] = ignore_mask[y1:y2+1, x1:x2+1].copy()
        crop_stuff[y_from:y_to+1, x_from:x_to+1] = stuff_mask[y1:y2+1, x1:x2+1].copy()

        joints = joints.astype('f')
        joints[:, :, :2] -= offset
        joints = joints.round().astype('i')
        return crop_img.astype('uint8'), crop_mask, crop_stuff, joints

    # distort image color
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

    def flip_img(self, img, mask, stuff_mask, joints):
        flipped_img = cv2.flip(img, 1)
        flipped_mask = cv2.flip(mask.astype(np.uint8), 1).astype('bool')
        stuff_mask = cv2.flip(stuff_mask, 1)
        joints[:, :, 0] = img.shape[1] - 1 - joints[:, :, 0]

        def swap_joints(joints, joint_type_1, joint_type_2):
            tmp = joints[:, joint_type_1, :].copy()
            joints[:, joint_type_1, :] = joints[:, joint_type_2, :]
            joints[:, joint_type_2, :] = tmp

        swap_joints(joints, JointType.LeftEye, JointType.RightEye)
        swap_joints(joints, JointType.LeftEar, JointType.RightEar)
        swap_joints(joints, JointType.LeftShoulder, JointType.RightShoulder)
        swap_joints(joints, JointType.LeftElbow, JointType.RightElbow)
        swap_joints(joints, JointType.LeftHand, JointType.RightHand)
        swap_joints(joints, JointType.LeftWaist, JointType.RightWaist)
        swap_joints(joints, JointType.LeftKnee, JointType.RightKnee)
        swap_joints(joints, JointType.LeftFoot, JointType.RightFoot)
        return flipped_img, flipped_mask, stuff_mask, joints

    def resize_data(self, img, ignore_mask, joints, stuff_mask, shape):
        """resize img and mask"""
        img_h, img_w, _ = img.shape

        resized_img = cv2.resize(img, shape)
        ignore_mask = cv2.resize(ignore_mask.astype(np.uint8), shape).astype('bool')
        joints[:, :, :2] = (joints[:, :, :2] * np.array(shape) / np.array((img_w, img_h))).astype(np.int64)
        stuff_mask = cv2.resize(stuff_mask, shape, interpolation=cv2.INTER_NEAREST)
        return resized_img, ignore_mask, joints, stuff_mask

    def get_joint_bboxes(self, joints):
        joint_bboxes = []
        for person_joints in joints:
            x1 = person_joints[person_joints[:, 2] > 0][:, 0].min()
            y1 = person_joints[person_joints[:, 2] > 0][:, 1].min()
            x2 = person_joints[person_joints[:, 2] > 0][:, 0].max()
            y2 = person_joints[person_joints[:, 2] > 0][:, 1].max()
            joint_bboxes.append([x1, y1, x2, y2])
        joint_bboxes = np.array(joint_bboxes)
        # for x1, y1, x2, y2 in joint_bboxes:  # draw joint bboxes
        #     cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        return joint_bboxes

    def augment_data(self, orig_img, ignore_mask, joints, stuff_mask):
        # TODO: need visialisation for debug
        aug_img = orig_img.copy()
        aug_img, ignore_mask, stuff_mask, joints = self.random_resize_img(
            aug_img, ignore_mask, stuff_mask, joints)
        aug_img, ignore_mask, stuff_mask, joints = self.random_rotate_img(
            aug_img, ignore_mask, stuff_mask, joints, params['max_rotate_degree'])
        aug_img, ignore_mask, stuff_mask, joints = self.random_crop_img(
            aug_img, ignore_mask, stuff_mask, joints)

        aug_img = self.distort_color(aug_img)

        if np.random.randint(2):
            aug_img, ignore_mask, stuff_mask, joints = self.flip_img(aug_img, ignore_mask, stuff_mask, joints)

        return aug_img, ignore_mask, joints, stuff_mask

    # return shape: (height, width)
    def generate_gaussian_heatmap(self, imshape, joint, sigma):
        x, y = joint
        grid_x = np.tile(np.arange(imshape[1]), (imshape[0], 1))
        grid_y = np.tile(np.arange(imshape[0]), (imshape[1], 1)).transpose()
        grid_distance = (grid_x - x) ** 2 + (grid_y - y) ** 2
        gaussian_heatmap = np.exp(-0.5 * grid_distance / sigma**2)
        return gaussian_heatmap

    def generate_heatmaps(self, img, joints, heatmap_sigma):
        heatmaps = np.zeros((0,) + img.shape[:-1])
        sum_heatmap = np.zeros(img.shape[:-1])
        for joint_index in range(len(JointType)):
            heatmap = np.zeros(img.shape[:-1])
            for person_joints in joints:
                if person_joints[joint_index, 2] > 0:
                    jointmap = self.generate_gaussian_heatmap(img.shape[:-1], person_joints[joint_index][:2], heatmap_sigma)
                    heatmap[jointmap > heatmap] = jointmap[jointmap > heatmap]
                    sum_heatmap[jointmap > sum_heatmap] = jointmap[jointmap > sum_heatmap]
            heatmaps = np.vstack((heatmaps, heatmap.reshape((1,) + heatmap.shape)))
        bg_heatmap = 1 - sum_heatmap  # background channel
        heatmaps = np.vstack((heatmaps, bg_heatmap[None]))
        return heatmaps.astype('f')

    # return shape: (2, height, width)
    def generate_constant_paf(self, imshape, joint_from, joint_to, paf_width):
        if np.array_equal(joint_from, joint_to): # same joint
            return np.zeros((2,) + imshape[:-1])

        joint_distance = np.linalg.norm(joint_to - joint_from)
        unit_vector = (joint_to - joint_from) / joint_distance
        rad = np.pi / 2
        rot_matrix = np.array([[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]])
        vertical_unit_vector = np.dot(rot_matrix, unit_vector)
        grid_x = np.tile(np.arange(imshape[1]), (imshape[0], 1))
        grid_y = np.tile(np.arange(imshape[0]), (imshape[1], 1)).transpose()
        horizontal_inner_product = unit_vector[0] * (grid_x - joint_from[0]) + unit_vector[1] * (grid_y - joint_from[1])
        horizontal_paf_flag = (0 <= horizontal_inner_product) & (horizontal_inner_product <= joint_distance)
        vertical_inner_product = vertical_unit_vector[0] * (grid_x - joint_from[0]) + vertical_unit_vector[1] * (grid_y - joint_from[1])
        vertical_paf_flag = np.abs(vertical_inner_product) <= paf_width
        paf_flag = horizontal_paf_flag & vertical_paf_flag
        constant_paf = np.stack((paf_flag, paf_flag)) * np.broadcast_to(unit_vector, imshape[:-1] + (2,)).transpose(2, 0, 1)
        return constant_paf

    def generate_pafs(self, img, joints, paf_sigma):
        pafs = np.zeros((0,) + img.shape[:-1])

        for limb in params['limbs_point']:
            paf = np.zeros((2,) + img.shape[:-1])
            paf_flags = np.zeros(paf.shape) # for constant paf

            for person_joints in joints:
                joint_from, joint_to = person_joints[limb]
                if joint_from[2] > 0 and joint_to[2] > 0:
                    limb_paf = self.generate_constant_paf(img.shape, joint_from[:2], joint_to[:2], paf_sigma)
                    limb_paf_flags = limb_paf != 0
                    paf_flags += np.broadcast_to(limb_paf_flags[0] | limb_paf_flags[1], limb_paf.shape)
                    paf += limb_paf

            paf[paf_flags > 0] /= paf_flags[paf_flags > 0]
            pafs = np.vstack((pafs, paf))
        return pafs.astype('f')

    def get_img_annotation(self, ind=None, img_id=None):
        """インデックスまたは img_id から coco annotation dataを抽出、条件に満たない場合はNoneを返す """
        annotations = None

        if ind is not None:
            img_id = self.imgIds[ind]
        anno_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)

        # annotation for that image
        if len(anno_ids) > 0:
            annotations_for_img = self.coco.loadAnns(anno_ids)

            person_cnt = 0
            valid_annotations_for_img = []
            for annotation in annotations_for_img:
                # if too few keypoints or too small
                if annotation['num_keypoints'] >= 5 and annotation['area'] > 32 * 32:
                    person_cnt += 1
                    valid_annotations_for_img.append(annotation)

            # if person annotation
            if person_cnt > 0:
                annotations = valid_annotations_for_img

        if self.mode == 'train':
            img_path = os.path.join(params['coco_dir'], 'train2017', self.coco.loadImgs([img_id])[0]['file_name'])
            mask_path = os.path.join(params['coco_dir'], 'ignore_mask_train2017', '{:012d}.png'.format(img_id))
        else:
            img_path = os.path.join(params['coco_dir'], 'val2017', self.coco.loadImgs([img_id])[0]['file_name'])
            mask_path = os.path.join(params['coco_dir'], 'ignore_mask_val2017', '{:012d}.png'.format(img_id))
        img = cv2.imread(img_path)
        ignore_mask = cv2.imread(mask_path, 0)
        if ignore_mask is None:
            ignore_mask = np.zeros(img.shape[:2], 'bool')
        else:
            ignore_mask = ignore_mask == 255

        masks = np.ones(img.shape[:2], 'uint8')
        for ann in annotations_for_img:
            mask = self.coco.annToMask(ann)
            if ann['iscrowd'] == 1:
                masks[mask == 1] = 0
            else:
                masks[mask == 1] = 2

        stuff_path = os.path.join(params['coco_stuff_dir'], 'annotations/COCO_train2014_{:012d}.mat'.format(img_id))
        if os.path.exists(stuff_path):
            stuff_mask = loadmat(stuff_path)['S']
        else:
            stuff_mask = np.zeros(ignore_mask.shape, 'uint8')
        if self.mode == 'eval':
            return img, img_id, annotations_for_img, ignore_mask, masks
        return img, img_id, annotations, ignore_mask, masks

    def parse_coco_annotation(self, img, annotations):
        """coco annotation dataのアノテーションをjoints配列に変換"""
        joints = np.zeros((0, len(JointType), 3), dtype=np.int32)

        for ann in annotations:
            person_joints = np.zeros((1, len(JointType), 3), dtype=np.int32)

            # convert joints position
            for i, joint_index in enumerate(params['coco_joint_indices']):
                person_joints[0][joint_index] = ann['keypoints'][i*3:i*3+3]

            # compute neck position
            if person_joints[0][JointType.LeftShoulder][2] > 0 and person_joints[0][JointType.RightShoulder][2] > 0:
                person_joints[0][JointType.Neck][0] = int((person_joints[0][JointType.LeftShoulder][0] + person_joints[0][JointType.RightShoulder][0]) / 2)
                person_joints[0][JointType.Neck][1] = int((person_joints[0][JointType.LeftShoulder][1] + person_joints[0][JointType.RightShoulder][1]) / 2)
                person_joints[0][JointType.Neck][2] = 2

            joints = np.vstack((joints, person_joints))
        return joints

    def generate_labels(self, img, annotations, ignore_mask, stuff_mask):
        joints = self.parse_coco_annotation(img, annotations)
        stuff_mask = stuff_mask.astype('i') - 1
        img, ignore_mask, joints, stuff_mask = self.augment_data(img, ignore_mask, joints, stuff_mask)
        resized_img, ignore_mask, resized_joints, resized_stuff = self.resize_data(img, ignore_mask, joints, stuff_mask, shape=(params['insize'], params['insize']))

        heatmaps = self.generate_heatmaps(resized_img, resized_joints, params['heatmap_sigma'])
        pafs = self.generate_pafs(resized_img, resized_joints, params['paf_sigma'])
        return resized_img, pafs, heatmaps, ignore_mask, resized_stuff

    def get_example(self, i):
        img, img_id, annotations, ignore_mask, stuff_mask = self.get_img_annotation(ind=i)

        if self.mode == 'eval':
            # don't need to make heatmaps/pafs
            return img, annotations, img_id

        # if no annotations are available
        while annotations is None:
            img_id = self.imgIds[np.random.randint(len(self))]
            img, img_id, annotations, ignore_mask, stuff_mask = self.get_img_annotation(img_id=img_id)

        resized_img, pafs, heatmaps, ignore_mask, stuff_mask = self.generate_labels(img, annotations, ignore_mask, stuff_mask)
        return resized_img, pafs, heatmaps, ignore_mask, stuff_mask

if __name__ == '__main__':
    mode = 'train'
    coco = COCO(os.path.join(params['coco_dir'], 'annotations/person_keypoints_{}2017.json'.format(mode)))
    data_loader = CocoDataLoader(coco, mode=mode)

    cv2.namedWindow('w', cv2.WINDOW_NORMAL)

    for i in range(len(data_loader)):
        orig_img, img_id, annotations, ignore_mask, stuff_mask = data_loader.get_img_annotation(ind=i)
        if annotations is not None:
            resized_img, pafs, heatmaps, ignore_mask, stuff_mask = data_loader.generate_labels(orig_img, annotations, ignore_mask, stuff_mask)

            # resize to view
            shape = (params['insize'],) * 2
            pafs = cv2.resize(pafs.transpose(1, 2, 0), shape).transpose(2, 0, 1)
            heatmaps = cv2.resize(heatmaps.transpose(1, 2, 0), shape).transpose(2, 0, 1)
            ignore_mask = cv2.resize(ignore_mask.astype(np.uint8)*255, shape) > 0
            stuff_mask = cv2.resize(stuff_mask, shape, interpolation=cv2.INTER_NEAREST)

            # view
            img = resized_img.copy()
            img = data_loader.overlay_pafs(img, pafs)
            img = data_loader.overlay_heatmap(img, heatmaps[:-1].max(axis=0))
            img = data_loader.overlay_ignore_mask(img, ignore_mask)
            img = data_loader.overlay_stuff_mask(img, stuff_mask, n_class=3)

            cv2.imshow('w', np.hstack((resized_img, img)))
            k = cv2.waitKey(0)
            if k == ord('q'):
                sys.exit()
            if k == ord('d'):
                import ipdb; ipdb.set_trace()
