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

        tmp_joints = np.ones((joints.shape[0], joints.shape[1], 3))
        tmp_joints[:, :, :2] = joints.copy()
        rotate_joints = np.dot(tmp_joints, R.T)  # apply rotation matrix to the joints
        return rotate_img, rotate_mask, rotate_stuff_mask, rotate_joints

    def compute_intersection(self, box1, box2):
        intersection_width =  np.minimum(box1[1][0], box2[1][0]) - np.maximum(box1[0][0], box2[0][0])
        intersection_height = np.minimum(box1[1][1], box2[1][1]) - np.maximum(box1[0][1], box2[0][1])

        if (intersection_height < 0) or (intersection_width < 0):
            return 0
        else:
            return intersection_width * intersection_height

    def compute_area(self, box):
        [[left, top], [right, bottom]] = box
        return (right - left) * (bottom - top)

    # intersection of box
    def compute_iob(self, box, joint_bbox):
        intersection = self.compute_intersection(box, joint_bbox)
        area = self.compute_area(joint_bbox)
        if area == 0:
            iob = 0
        else:
            iob = intersection / area
        return iob

    def validate_crop_area(self, crop_bbox, joint_bboxes, iob_thresh):
        valid_iob_list = []
        iob_list = []
        for joint_bbox in joint_bboxes:
            iob = self.compute_iob(crop_bbox, joint_bbox)
            valid_iob_list.append(iob <= 0 or iob >= iob_thresh)
            iob_list.append(iob)
        return valid_iob_list, np.array(iob_list)

    def random_crop_img(self, orig_img, ignore_mask, stuff_mask, joints, valid_joints, joint_bboxes, min_crop_size):
        # get correct crop area
        iteration = 0
        while True:
            iteration += 1
            crop_width = crop_height = np.random.randint(min_crop_size, min(orig_img.shape[:-1]) + 1)
            crop_left = np.random.randint(orig_img.shape[1] - crop_width + 1)
            crop_top = np.random.randint(orig_img.shape[0] - crop_height + 1)
            crop_right = crop_left + crop_width
            crop_bottom = crop_top + crop_height

            valid_iob_list, iob_list = self.validate_crop_area([[crop_left, crop_top], [crop_right, crop_bottom]], joint_bboxes, params['crop_iob_thresh'])
            if np.all(valid_iob_list) or iteration > 10:
                break

        cropped_img = orig_img[crop_top:crop_bottom, crop_left:crop_right]
        ignore_mask = ignore_mask[crop_top:crop_bottom, crop_left:crop_right]
        stuff_mask = stuff_mask[crop_top:crop_bottom, crop_left:crop_right]
        joints[:, :, 0][valid_joints] -= crop_left
        joints[:, :, 1][valid_joints] -= crop_top
        valid_joints[iob_list == 0, :] = False

        return cropped_img, ignore_mask, stuff_mask, joints, valid_joints

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

    def resize_data(self, img, ignore_mask, joints, stuff_mask, shape):
        """resize img and mask"""
        img_h, img_w, _ = img.shape

        resized_img = cv2.resize(img, shape)
        ignore_mask = cv2.resize(ignore_mask.astype(np.uint8), shape).astype('bool')
        joints = (joints * np.array(shape) / np.array((img_w, img_h))).astype(np.int64)
        stuff_mask = cv2.resize(stuff_mask, shape, interpolation=cv2.INTER_NEAREST)
        return resized_img, ignore_mask, joints, stuff_mask

    def flip_img(self, img, mask, stuff_mask, joints, valid_joints):
        flipped_img = cv2.flip(img, 1)
        flipped_mask = cv2.flip(mask.astype(np.uint8), 1).astype('bool')
        stuff_mask = cv2.flip(stuff_mask, 1)
        joints[:, :, 0] = img.shape[1] - 1 - joints[:, :, 0]

        def swap_joints(joints, valid_joints, joint_type_1, joint_type_2):
            tmp = joints[:, joint_type_1, :].copy()
            joints[:, joint_type_1, :] = joints[:, joint_type_2, :]
            joints[:, joint_type_2, :] = tmp

            tmp = valid_joints[:, joint_type_1].copy()
            valid_joints[:, joint_type_1] = valid_joints[:, joint_type_2]
            valid_joints[:, joint_type_2] = tmp

        swap_joints(joints, valid_joints, JointType.LeftEye, JointType.RightEye)
        swap_joints(joints, valid_joints, JointType.LeftEar, JointType.RightEar)
        swap_joints(joints, valid_joints, JointType.LeftShoulder, JointType.RightShoulder)
        swap_joints(joints, valid_joints, JointType.LeftElbow, JointType.RightElbow)
        swap_joints(joints, valid_joints, JointType.LeftHand, JointType.RightHand)
        swap_joints(joints, valid_joints, JointType.LeftWaist, JointType.RightWaist)
        swap_joints(joints, valid_joints, JointType.LeftKnee, JointType.RightKnee)
        swap_joints(joints, valid_joints, JointType.LeftFoot, JointType.RightFoot)

        return flipped_img, flipped_mask, stuff_mask, joints, valid_joints

    def augment_data(self, orig_img, ignore_mask, joints, valid_joints, stuff_mask, joint_bboxes, min_crop_size):
        """augment data"""
        aug_img = orig_img.copy()

        aug_img, ignore_mask, stuff_mask, joints = self.random_rotate_img(
            aug_img, ignore_mask, stuff_mask, joints, params['max_rotate_degree'])

        box_sizes = np.linalg.norm(joint_bboxes[:, 1] - joint_bboxes[:, 0], axis=1)
        min_crop_size = np.min((min(orig_img.shape[:-1]), min_crop_size, int(box_sizes.min() * 5)))
        aug_img, ignore_mask, stuff_mask, joints, valid_joints = self.random_crop_img(
            aug_img, ignore_mask, stuff_mask, joints, valid_joints, joint_bboxes, min_crop_size)

        # distort color
        aug_img = self.distort_color(aug_img)

        # flip image
        if np.random.randint(2):
            aug_img, ignore_mask, stuff_mask, joints, valid_joints = self.flip_img(
                aug_img, ignore_mask, stuff_mask, joints, valid_joints)

        return aug_img, ignore_mask, joints, valid_joints, stuff_mask

    # return shape: (height, width)
    def gen_gaussian_heatmap(self, imshape, joint, sigma):
        x, y = joint
        grid_x = np.tile(np.arange(imshape[1]), (imshape[0], 1))
        grid_y = np.tile(np.arange(imshape[0]), (imshape[1], 1)).transpose()
        grid_distance = (grid_x - x) ** 2 + (grid_y - y) ** 2
        gaussian_heatmap = np.exp(-0.5 * grid_distance / sigma**2)
        return gaussian_heatmap

    def compute_heatmaps(self, img, joints, valid_joints, heatmap_sigma):
        """compute heatmaps"""
        heatmaps = np.zeros((0,) + img.shape[:-1])
        sum_heatmap = np.zeros(img.shape[:-1])

        for joint_index in range(len(JointType)):
            heatmap = np.zeros(img.shape[:-1])
            for person_index, person_joints in enumerate(joints):
                if valid_joints[person_index][joint_index]:
                    jointmap = self.gen_gaussian_heatmap(img.shape[:-1], person_joints[joint_index], heatmap_sigma)
                    heatmap[jointmap > heatmap] = jointmap[jointmap > heatmap]
                    sum_heatmap[jointmap > sum_heatmap] = jointmap[jointmap > sum_heatmap]
            heatmaps = np.vstack((heatmaps, heatmap.reshape((1,) + heatmap.shape)))
        bg_heatmap = 1 - sum_heatmap  # background channel
        heatmaps = np.vstack((heatmaps, bg_heatmap[None]))
        return heatmaps.astype('f')

    # return shape: (2, height, width)
    def gen_constant_paf(self, imshape, joint_from, joint_to, paf_width):
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

    def compute_pafs(self, img, joints, valid_joints, paf_sigma):
        """compute pafs"""
        pafs = np.zeros((0,) + img.shape[:-1])

        for limb in params['limbs_point']:
            paf = np.zeros((2,) + img.shape[:-1])
            paf_flags = np.zeros(paf.shape) # for constant paf

            for person_index, person_joints in enumerate(joints):
                if valid_joints[person_index][limb[0]] and valid_joints[person_index][limb[1]]:
                    limb_paf = self.gen_constant_paf(img.shape, np.array(person_joints[limb[0]]), np.array(person_joints[limb[1]]), paf_sigma)
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
        joints = np.zeros((0, len(JointType), 2), dtype=np.int32)
        valid_joints = np.zeros((0, len(JointType)), dtype=np.bool)
        joint_bboxes = np.zeros((0, 2, 2), np.int32)

        for ann in annotations:
            person_joints = np.zeros((1, len(JointType), 2), dtype=np.int32)
            person_valid_joints = np.zeros((1, len(JointType)), dtype=np.bool)
            person_joint_bbox = np.array([[[np.iinfo(np.int32).max, np.iinfo(np.int32).max], [np.iinfo(np.int32).min, np.iinfo(np.int32).min]]], np.int32)

            # convert joints position
            for i, joint_index in enumerate(params['coco_joint_indices']):
                valid_joint = bool(ann['keypoints'][i * 3 + 2])
                if valid_joint:
                    person_valid_joints[0][joint_index] = True
                    person_joints[0][joint_index][0] = ann['keypoints'][i * 3]
                    person_joints[0][joint_index][1] = ann['keypoints'][i * 3 + 1]
                    person_joint_bbox[0][0][0] = np.minimum(person_joint_bbox[0][0][0],  person_joints[0][joint_index][0]) # left
                    person_joint_bbox[0][0][1] = np.minimum(person_joint_bbox[0][0][1],  person_joints[0][joint_index][1]) # top
                    person_joint_bbox[0][1][0] = np.maximum(person_joint_bbox[0][1][0],  person_joints[0][joint_index][0]) # right
                    person_joint_bbox[0][1][1] = np.maximum(person_joint_bbox[0][1][1],  person_joints[0][joint_index][1]) # bottom

            # compute neck position
            if bool(person_valid_joints[0][JointType.LeftShoulder]) and bool(person_valid_joints[0][JointType.RightShoulder]):
                person_valid_joints[0][JointType.Neck] = True
                person_joints[0][JointType.Neck][0] = int((person_joints[0][JointType.LeftShoulder][0] + person_joints[0][JointType.RightShoulder][0]) / 2)
                person_joints[0][JointType.Neck][1] = int((person_joints[0][JointType.LeftShoulder][1] + person_joints[0][JointType.RightShoulder][1]) / 2)

            joints = np.vstack((joints, person_joints))
            valid_joints = np.vstack((valid_joints, person_valid_joints))
            joint_bboxes = np.vstack((joint_bboxes, person_joint_bbox))

        return joints, valid_joints, joint_bboxes

    def generate_labels(self, img, annotations, ignore_mask, stuff_mask):
        joints, valid_joints, joint_bboxes = self.parse_coco_annotation(img, annotations)
        stuff_mask = stuff_mask.astype('i') - 1
        if self.mode != 'eval':
            img, ignore_mask, joints, valid_joints, stuff_mask = self.augment_data(img, ignore_mask, joints, valid_joints, stuff_mask, joint_bboxes, params['crop_size'])
        resized_img, ignore_mask, resized_joints, resized_stuff = self.resize_data(img, ignore_mask, joints, stuff_mask, shape=(params['insize'], params['insize']))

        heatmaps = self.compute_heatmaps(resized_img, resized_joints, valid_joints, params['heatmap_sigma'])
        pafs = self.compute_pafs(resized_img, resized_joints, valid_joints, params['paf_sigma'])
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
    count = 0
    for i in range(len(data_loader)):
        orig_img, img_id, annotations, ignore_mask, stuff_mask = data_loader.get_img_annotation(ind=random.randint(0, len(data_loader)))
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
            # img = data_loader.overlay_stuff_mask(img, stuff_mask, n_class=3)

            cv2.imshow('w', np.hstack((resized_img, img)))
            k = cv2.waitKey(1)
            if k == ord('q'):
                sys.exit()
