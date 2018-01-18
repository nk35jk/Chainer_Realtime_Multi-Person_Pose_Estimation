import time
import numpy as np
import chainer
from chainer import functions as F, links as L
from chainer.links.caffe import CaffeFunction


def copy_mobilenet_params(model):
    print('Copying params of pretrained model...')
    layer_names = [
        '', '', ...
    ]
    pre_model = CaffeFunction('models/mobilenet.caffemodel')
    for layer_name in layer_names:
        if len(layer_name.split('/')) > 1:
            exec('model.{}.{}.copyparams(pre_model["{}"])'.format(layer_name.split('/')[0], layer_name.split('/')[1], layer_name))
        else:
            exec('model.{}.copyparams(pre_model["{}"])'.format(layer_name, layer_name))
    print('Done.')


class MobileNet(chainer.Chain):
    """MobileNet"""

    insize = 368
    downscale = 8
    pad = downscale

    mean = [103.94,116.78,123.68]
    scale = 0.017

    def __init__(self, joints=19, limbs=38, compute_mask=False):
        super(MobileNet, self).__init__()
        self.joints = joints
        self.limbs = limbs
        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 32, 3, stride=2, pad=1, nobias=True)  # 1/2
            self.conv1_bn = L.BatchNormalization(32)

            self.conv2_1_dw = L.DepthwiseConvolution2D(32, 1, 3, stride=1, pad=1, nobias=True)
            self.conv2_1_dw_bn = L.BatchNormalization(32)
            self.conv2_1_sp = L.Convolution2D(32, 64, 1, nobias=True)
            self.conv2_1_sp_bn = L.BatchNormalization(64)
            self.conv2_2_dw = L.DepthwiseConvolution2D(64, 1, 3, stride=2, pad=1, nobias=True)  # 1/4
            self.conv2_2_dw_bn = L.BatchNormalization(64)
            self.conv2_2_sp = L.Convolution2D(64, 128, 1, nobias=True)
            self.conv2_2_sp_bn = L.BatchNormalization(128)

            self.conv3_1_dw = L.DepthwiseConvolution2D(128, 1, 3, stride=1, pad=1, nobias=True)
            self.conv3_1_dw_bn = L.BatchNormalization(128)
            self.conv3_1_sp = L.Convolution2D(128, 128, 1, nobias=True)
            self.conv3_1_sp_bn = L.BatchNormalization(128)
            self.conv3_2_dw = L.DepthwiseConvolution2D(128, 1, 3, stride=2, pad=1, nobias=True)  # 1/8
            self.conv3_2_dw_bn = L.BatchNormalization(128)
            self.conv3_2_sp = L.Convolution2D(128, 256, 1, nobias=True)
            self.conv3_2_sp_bn = L.BatchNormalization(256)

            self.conv4_1_dw = L.DepthwiseConvolution2D(256, 1, 3, stride=1, pad=1, nobias=True)
            self.conv4_1_dw_bn = L.BatchNormalization(256)
            self.conv4_1_sp = L.Convolution2D(256, 256, 1, nobias=True)
            self.conv4_1_sp_bn = L.BatchNormalization(256)
            self.conv4_2_dw = L.DepthwiseConvolution2D(256, 1, 3, stride=1, pad=1, nobias=True)  # stride=2
            self.conv4_2_dw_bn = L.BatchNormalization(256)
            self.conv4_2_sp = L.Convolution2D(256, 512, 1, nobias=True)
            self.conv4_2_sp_bn = L.BatchNormalization(512)

            self.conv5_1_dw = L.DepthwiseConvolution2D(512, 1, 3, stride=1, pad=1, nobias=True)
            self.conv5_1_dw_bn = L.BatchNormalization(512)
            self.conv5_1_sp = L.Convolution2D(512, 512, 1, nobias=True)
            self.conv5_1_sp_bn = L.BatchNormalization(512)
            self.conv5_2_dw = L.DepthwiseConvolution2D(512, 1, 3, stride=1, pad=1, nobias=True)
            self.conv5_2_dw_bn = L.BatchNormalization(512)
            self.conv5_2_sp = L.Convolution2D(512, 512, 1, nobias=True)
            self.conv5_2_sp_bn = L.BatchNormalization(512)
            self.conv5_3_dw = L.DepthwiseConvolution2D(512, 1, 3, stride=1, pad=1, nobias=True)
            self.conv5_3_dw_bn = L.BatchNormalization(512)
            self.conv5_3_sp = L.Convolution2D(512, 512, 1, nobias=True)
            self.conv5_3_sp_bn = L.BatchNormalization(512)
            self.conv5_4_dw = L.DepthwiseConvolution2D(512, 1, 3, stride=1, pad=1, nobias=True)
            self.conv5_4_dw_bn = L.BatchNormalization(512)
            self.conv5_4_sp = L.Convolution2D(512, 512, 1, nobias=True)
            self.conv5_4_sp_bn = L.BatchNormalization(512)
            self.conv5_5_dw = L.DepthwiseConvolution2D(512, 1, 3, stride=1, pad=1, nobias=True)
            self.conv5_5_dw_bn = L.BatchNormalization(512)
            self.conv5_5_sp = L.Convolution2D(512, 512, 1, nobias=True)
            self.conv5_5_sp_bn = L.BatchNormalization(512)
            self.conv5_6_dw = L.DepthwiseConvolution2D(512, 1, 3, stride=1, pad=1, nobias=True)  # stride=2
            self.conv5_6_dw_bn = L.BatchNormalization(512)
            self.conv5_6_sp = L.Convolution2D(512, 1024, 1, nobias=True)
            self.conv5_6_sp_bn = L.BatchNormalization(1024)

            self.conv6_dw = L.DepthwiseConvolution2D(1024, 1, 3, stride=1, pad=1, nobias=True)
            self.conv6_dw_bn = L.BatchNormalization(1024)
            self.conv6_sp = L.Convolution2D(1024, 1024, 1, nobias=True)
            self.conv6_sp_bn = L.BatchNormalization(1024)

            self.conv7_dw = L.DepthwiseConvolution2D(1024, 1, 3, stride=1, pad=1, nobias=True)
            self.conv7_dw_bn = L.BatchNormalization(1024)
            self.conv7_sp = L.Convolution2D(1024, limbs+joints, 1, nobias=True)

            # self.fc7 = L.Linear(1024, 1000)

    def __call__(self, x):
        pafs, heatmaps = [], []

        h = F.relu(self.conv1_bn(self.conv1(x)))

        h = F.relu(self.conv2_1_dw_bn(self.conv2_1_dw(h)))
        h = F.relu(self.conv2_1_sp_bn(self.conv2_1_sp(h)))
        h = F.relu(self.conv2_2_dw_bn(self.conv2_2_dw(h)))
        h = F.relu(self.conv2_2_sp_bn(self.conv2_2_sp(h)))

        h = F.relu(self.conv3_1_dw_bn(self.conv3_1_dw(h)))
        h = F.relu(self.conv3_1_sp_bn(self.conv3_1_sp(h)))
        h = F.relu(self.conv3_2_dw_bn(self.conv3_2_dw(h)))
        h = F.relu(self.conv3_2_sp_bn(self.conv3_2_sp(h)))

        h = F.relu(self.conv4_1_dw_bn(self.conv4_1_dw(h)))
        h = F.relu(self.conv4_1_sp_bn(self.conv4_1_sp(h)))
        h = F.relu(self.conv4_2_dw_bn(self.conv4_2_dw(h)))
        h = F.relu(self.conv4_2_sp_bn(self.conv4_2_sp(h)))

        h = F.relu(self.conv5_1_dw_bn(self.conv5_1_dw(h)))
        h = F.relu(self.conv5_1_sp_bn(self.conv5_1_sp(h)))
        h = F.relu(self.conv5_2_dw_bn(self.conv5_2_dw(h)))
        h = F.relu(self.conv5_2_sp_bn(self.conv5_2_sp(h)))
        h = F.relu(self.conv5_3_dw_bn(self.conv5_3_dw(h)))
        h = F.relu(self.conv5_3_sp_bn(self.conv5_3_sp(h)))
        h = F.relu(self.conv5_4_dw_bn(self.conv5_4_dw(h)))
        h = F.relu(self.conv5_4_sp_bn(self.conv5_4_sp(h)))
        h = F.relu(self.conv5_5_dw_bn(self.conv5_5_dw(h)))
        h = F.relu(self.conv5_5_sp_bn(self.conv5_5_sp(h)))
        h = F.relu(self.conv5_6_dw_bn(self.conv5_6_dw(h)))
        h = F.relu(self.conv5_6_sp_bn(self.conv5_6_sp(h)))

        h = F.relu(self.conv6_dw_bn(self.conv6_dw(h)))
        h = F.relu(self.conv6_sp_bn(self.conv6_sp(h)))

        h = F.relu(self.conv7_dw_bn(self.conv7_dw(h)))
        h = self.conv7_sp(h)

        pafs.append(h[:, :self.limbs])
        heatmaps.append(h[:, -self.joints:])

        # h = F.average_pooling_2d(h, 7)
        # h = self.fc7(h)

        return pafs, heatmaps

if __name__ == '__main__':
    model = MobileNet()
    arr = np.random.rand(1, 3, 368, 368).astype('f')
    y = model(arr)
