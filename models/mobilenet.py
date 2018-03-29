import time
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L


class MobileNet(chainer.Chain):
    """MobileNet"""

    insize = 368
    downscale = pad = 8

    def __init__(self, joints=19, limbs=38):
        super(MobileNet, self).__init__()
        self.joints = joints
        self.limbs = limbs
        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 32, 3, stride=2, pad=1, nobias=True)
            self.conv1_bn = L.BatchNormalization(32)

            self.conv2_1_dw = L.DepthwiseConvolution2D(32, 1, 3, stride=1, pad=1, nobias=True)
            self.conv2_1_dw_bn = L.BatchNormalization(32)
            self.conv2_1_sp = L.Convolution2D(32, 64, 1, nobias=True)
            self.conv2_1_sp_bn = L.BatchNormalization(64)
            self.conv2_2_dw = L.DepthwiseConvolution2D(64, 1, 3, stride=2, pad=1, nobias=True)
            self.conv2_2_dw_bn = L.BatchNormalization(64)
            self.conv2_2_sp = L.Convolution2D(64, 128, 1, nobias=True)
            self.conv2_2_sp_bn = L.BatchNormalization(128)

            self.conv3_1_dw = L.DepthwiseConvolution2D(128, 1, 3, stride=1, pad=1, nobias=True)
            self.conv3_1_dw_bn = L.BatchNormalization(128)
            self.conv3_1_sp = L.Convolution2D(128, 128, 1, nobias=True)
            self.conv3_1_sp_bn = L.BatchNormalization(128)
            self.conv3_2_dw = L.DepthwiseConvolution2D(128, 1, 3, stride=2, pad=1, nobias=True)
            self.conv3_2_dw_bn = L.BatchNormalization(128)
            self.conv3_2_sp = L.Convolution2D(128, 128, 1, nobias=True)
            self.conv3_2_sp_bn = L.BatchNormalization(128)

            self.conv4_1_dw = L.DepthwiseConvolution2D(128, 1, 5, stride=1, pad=2, nobias=True)
            self.conv4_1_dw_bn = L.BatchNormalization(128)
            self.conv4_1_sp = L.Convolution2D(128, 128, 1, nobias=True)
            self.conv4_1_sp_bn = L.BatchNormalization(128)
            self.conv4_2_dw = L.DepthwiseConvolution2D(128, 1, 5, stride=1, pad=2, nobias=True)  # stride=2
            self.conv4_2_dw_bn = L.BatchNormalization(128)
            self.conv4_2_sp = L.Convolution2D(128, 256, 1, nobias=True)
            self.conv4_2_sp_bn = L.BatchNormalization(256)

            self.conv5_1_dw = L.DepthwiseConvolution2D(256, 1, 5, stride=1, pad=2, nobias=True)
            self.conv5_1_dw_bn = L.BatchNormalization(256)
            self.conv5_1_sp = L.Convolution2D(256, 256, 1, nobias=True)
            self.conv5_1_sp_bn = L.BatchNormalization(256)
            self.conv5_2_dw = L.DepthwiseConvolution2D(256, 1, 5, stride=1, pad=2, nobias=True)
            self.conv5_2_dw_bn = L.BatchNormalization(256)
            self.conv5_2_sp = L.Convolution2D(256, 256, 1, nobias=True)
            self.conv5_2_sp_bn = L.BatchNormalization(256)
            self.conv5_3_dw = L.DepthwiseConvolution2D(256, 1, 5, stride=1, pad=2, nobias=True)
            self.conv5_3_dw_bn = L.BatchNormalization(256)
            self.conv5_3_sp = L.Convolution2D(256, 256, 1, nobias=True)
            self.conv5_3_sp_bn = L.BatchNormalization(256)
            self.conv5_4_dw = L.DepthwiseConvolution2D(256, 1, 5, stride=1, pad=2, nobias=True)
            self.conv5_4_dw_bn = L.BatchNormalization(256)
            self.conv5_4_sp = L.Convolution2D(256, 256, 1, nobias=True)
            self.conv5_4_sp_bn = L.BatchNormalization(256)
            self.conv5_5_dw = L.DepthwiseConvolution2D(256, 1, 5, stride=1, pad=2, nobias=True)
            self.conv5_5_dw_bn = L.BatchNormalization(256)
            self.conv5_5_sp = L.Convolution2D(256, 256, 1, nobias=True)
            self.conv5_5_sp_bn = L.BatchNormalization(256)
            self.conv5_6_dw = L.DepthwiseConvolution2D(256, 1, 5, stride=1, pad=2, nobias=True)  # stride=2
            self.conv5_6_dw_bn = L.BatchNormalization(256)
            self.conv5_6_sp = L.Convolution2D(256, 256, 1, nobias=True)
            self.conv5_6_sp_bn = L.BatchNormalization(256)

            self.conv6_dw = L.DepthwiseConvolution2D(256, 1, 5, stride=1, pad=2, nobias=True)
            self.conv6_dw_bn = L.BatchNormalization(256)
            self.conv6_sp = L.Convolution2D(256, 256, 1, nobias=True)
            self.conv6_sp_bn = L.BatchNormalization(256)

            self.conv7 = L.Convolution2D(256, limbs+joints, 1)

            # self.fc7 = L.Linear(1024, 1000)

    def __call__(self, x):
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

        h = self.conv7(h)

        pafs = [h[:, :self.limbs]]
        heatmaps = [h[:, -self.joints:]]

        # h = F.average_pooling_2d(h, 7)
        # h = self.fc7(h)

        return pafs, heatmaps

if __name__ == '__main__':
    chainer.config.enable_backprop = False
    chainer.config.train = False

    model = MobileNet()
    arr = np.random.randn(1, 3, model.insize, model.insize).astype('f')
    h1s, h2s = model(arr)
