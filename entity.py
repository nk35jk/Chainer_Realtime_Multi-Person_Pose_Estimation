import argparse
from enum import IntEnum

from models.CocoPoseNet import CocoPoseNet
from models.posenet import PoseNet
from models.nn1 import NN1
from models.resnetfpn import ResNetFPN

from models.FaceNet import FaceNet
from models.HandNet import HandNet


class JointType(IntEnum):
    """関節の種類を表す """
    Nose = 0
    """ 鼻 """
    Neck = 1
    """ 首 """
    RightShoulder = 2
    """ 右肩 """
    RightElbow = 3
    """ 右肘 """
    RightHand = 4
    """ 右手 """
    LeftShoulder = 5
    """ 左肩 """
    LeftElbow = 6
    """ 左肘 """
    LeftHand = 7
    """ 左手 """
    RightWaist = 8
    """ 右腰 """
    RightKnee = 9
    """ 右膝 """
    RightFoot = 10
    """ 右足 """
    LeftWaist = 11
    """ 左腰 """
    LeftKnee = 12
    """ 左膝 """
    LeftFoot = 13
    """ 左足 """
    RightEye = 14
    """ 右目 """
    LeftEye = 15
    """ 左目 """
    RightEar = 16
    """ 右耳 """
    LeftEar = 17
    """ 左耳 """

params = {
    'coco_dir': 'coco',
    'coco_stuff_dir': 'cocostuff-10k-v1',
    'archs': {
        # 'posenet': CocoPoseNet,
        'posenet': PoseNet,
        'facenet': FaceNet,
        'handnet': HandNet,
        'nn1': NN1,
        'resnetfpn': ResNetFPN,
    },
    # training params
    'insize': 368,
    'downscale': 8,
    'paf_sigma': 8,
    'heatmap_sigma': 7,
    'target_dist': 0.6,
    'scale_min': 0.5,
    'scale_max': 1.1,
    'max_rotate_degree': 40,
    'center_perterb_max': 40,
    'mask_loss_ratio': 0.005,

    # inference params
    'inference_img_size': 368,
    # 'inference_scales': [0.5, 1, 1.5, 2],
    'inference_scales': [1.0],
    'heatmap_size': 320,
    'gaussian_sigma': 2.5,
    'ksize': 17,
    'n_integ_points': 10,       # 1つのconnectionを10等分して積分計算
    'n_integ_points_thresh': 8, # 1つのconnectionで最低8点以上が閾値を超えた場合に有効
    'heatmap_peak_thresh': 0.1,
    'inner_product_thresh': 0.05,
    'length_penalty_ratio': 0.5,
    'n_subset_limbs_thresh': 7,
    'subset_score_thresh': 0.4,
    'limbs_point': [
        [JointType.Neck, JointType.RightWaist],
        [JointType.RightWaist, JointType.RightKnee],
        [JointType.RightKnee, JointType.RightFoot],
        [JointType.Neck, JointType.LeftWaist],
        [JointType.LeftWaist, JointType.LeftKnee],
        [JointType.LeftKnee, JointType.LeftFoot],
        [JointType.Neck, JointType.RightShoulder],
        [JointType.RightShoulder, JointType.RightElbow],
        [JointType.RightElbow, JointType.RightHand],
        [JointType.RightShoulder, JointType.RightEar],
        [JointType.Neck, JointType.LeftShoulder],
        [JointType.LeftShoulder, JointType.LeftElbow],
        [JointType.LeftElbow, JointType.LeftHand],
        [JointType.LeftShoulder, JointType.LeftEar],
        [JointType.Neck, JointType.Nose],
        [JointType.Nose, JointType.RightEye],
        [JointType.Nose, JointType.LeftEye],
        [JointType.RightEye, JointType.RightEar],
        [JointType.LeftEye, JointType.LeftEar]
    ],
    'coco_joint_indices': [
        JointType.Nose,
        JointType.LeftEye,
        JointType.RightEye,
        JointType.LeftEar,
        JointType.RightEar,
        JointType.LeftShoulder,
        JointType.RightShoulder,
        JointType.LeftElbow,
        JointType.RightElbow,
        JointType.LeftHand,
        JointType.RightHand,
        JointType.LeftWaist,
        JointType.RightWaist,
        JointType.LeftKnee,
        JointType.RightKnee,
        JointType.LeftFoot,
        JointType.RightFoot
    ],

    # face params
    'face_inference_img_size': 368,
    'face_heatmap_peak_thresh': 0.1,
    'face_crop_scale': 1.5,
    'face_line_indices': [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], # 輪郭
        [17, 18], [18, 19], [19, 20], [20, 21], # 右眉
        [22, 23], [23, 24], [24, 25], [25, 26], # 左眉
        [27, 28], [28, 29], [29, 30], # 鼻
        [31, 32], [32, 33], [33, 34], [34, 35], # 鼻下の横線
        [36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [41, 36], # 右目
        [42, 43], [43, 44], [44, 45], [45, 46], [46, 47], [47, 42], # 左目
        [48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54], [54, 55], [55, 56], [56, 57], [57, 58], [58, 59], [59, 48], # 唇外輪
        [60, 61], [61, 62], [62, 63], [63, 64], [64, 65], [65, 66], [66, 67], [67, 60] # 唇内輪
    ],

    # hand params
    'hand_inference_img_size': 368,
    'hand_heatmap_peak_thresh': 0.1,
    'fingers_indices': [
        [[0, 1], [1, 2], [2, 3], [3, 4]],
        [[0, 5], [5, 6], [6, 7], [7, 8]],
        [[0, 9], [9, 10], [10, 11], [11, 12]],
        [[0, 13], [13, 14], [14, 15], [15, 16]],
        [[0, 17], [17, 18], [18, 19], [19, 20]],
    ],
}


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
    parser.add_argument('--eval_samples', type=int, default=40,
                        help='Number of validation samples')
    parser.add_argument('--iteration', '-i', type=int, default=600000,
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
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--mask', action='store_true')
    parser.set_defaults(test=False)
    parser.set_defaults(mask=False)
    args = parser.parse_args()
    params['insize'] = params['archs'][args.arch].insize
    params['downscale'] = params['archs'][args.arch].downscale
    return args
