import time
import numpy as np
import chainer
from chainer import functions as F, links as L
from chainer import initializers


class BottleNeckA(chainer.Chain):
    def __init__(self, in_size, ch, out_size, stride=2):
        super(BottleNeckA, self).__init__()
        initialW = initializers.HeNormal()

        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_size, ch, 1, stride, 0, initialW=initialW, nobias=True)
            self.bn1 = L.BatchNormalization(ch)
            self.conv2 = L.Convolution2D(
                ch, ch, 3, 1, 1, initialW=initialW, nobias=True)
            self.bn2 = L.BatchNormalization(ch)
            self.conv3 = L.Convolution2D(
                ch, out_size, 1, 1, 0, initialW=initialW, nobias=True)
            self.bn3 = L.BatchNormalization(out_size)

            self.conv4 = L.Convolution2D(
                in_size, out_size, 1, stride, 0,
                initialW=initialW, nobias=True)
            self.bn4 = L.BatchNormalization(out_size)

    def __call__(self, x):
        h1 = F.relu(self.bn1(self.conv1(x)))
        h1 = F.relu(self.bn2(self.conv2(h1)))
        h1 = self.bn3(self.conv3(h1))
        h2 = self.bn4(self.conv4(x))
        return F.relu(h1 + h2)


class BottleNeckB(chainer.Chain):
    def __init__(self, in_size, ch):
        super(BottleNeckB, self).__init__()
        initialW = initializers.HeNormal()

        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_size, ch, 1, 1, 0, initialW=initialW, nobias=True)
            self.bn1 = L.BatchNormalization(ch)
            self.conv2 = L.Convolution2D(
                ch, ch, 3, 1, 1, initialW=initialW, nobias=True)
            self.bn2 = L.BatchNormalization(ch)
            self.conv3 = L.Convolution2D(
                ch, in_size, 1, 1, 0, initialW=initialW, nobias=True)
            self.bn3 = L.BatchNormalization(in_size)

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))
        return F.relu(h + x)


class Block(chainer.Chain):
    def __init__(self, layer, in_size, ch, out_size, stride=2):
        super(Block, self).__init__()
        self.add_link('a', BottleNeckA(in_size, ch, out_size, stride))
        for i in range(1, layer):
            self.add_link('b{}'.format(i), BottleNeckB(out_size, ch))
        self.layer = layer

    def __call__(self, x):
        h = self.a(x)
        for i in range(1, self.layer):
            h = self['b{}'.format(i)](h)
        return h


class ResNet(chainer.Chain):
    def __init__(self):
        super(ResNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                3, 64, 7, 2, 3, initialW=initializers.HeNormal(), nobias=True)
            self.bn1 = L.BatchNormalization(64)
            self.res2 = Block(3, 64, 64, 256, 1)
            self.res3 = Block(4, 256, 128, 512)
            self.res4 = Block(6, 512, 256, 1024)
            self.res5 = Block(3, 1024, 512, 2048)

    def __call__(self, x):
        hs = []
        h = self.bn1(self.conv1(x))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = self.res2(h)
        hs.append(h)
        h = self.res3(h)
        hs.append(h)
        h = self.res4(h)
        hs.append(h)
        h = self.res5(h)
        hs.append(h)
        return hs


class ResNet50(chainer.Chain):
    """resnet50 FPN"""

    insize = 368

    def __init__(self, joints=19, limbs=38):
        super(ResNet50, self).__init__()
        with self.init_scope():
            self.res = ResNet()
            self.newC5 = L.Convolution2D(2048, 256, 1, stride=1, pad=0)
            self.newC4 = L.Convolution2D(1024, 256, 1, stride=1, pad=0)
            self.newC3 = L.Convolution2D(512, 256, 1, stride=1, pad=0)
            self.newC2 = L.Convolution2D(256, 256, 1, stride=1, pad=0)

            self.upP5 = L.Deconvolution2D(256, 256, 4, stride=2, pad=1)
            self.upP4 = L.Deconvolution2D(256, 256, 4, stride=2, pad=1)
            self.upP3 = L.Deconvolution2D(256, 256, 4, stride=2, pad=1)

            self.conv1_L1 = L.Convolution2D(256, 256, 3, stride=1, pad=1)
            self.conv2_L1 = L.Convolution2D(256, 256, 3, stride=1, pad=1)
            self.conv3_L1 = L.Convolution2D(256, 256, 3, stride=1, pad=1)
            self.conv4_L1 = L.Convolution2D(256, limbs, 1, stride=1, pad=0)
            self.conv1_L2 = L.Convolution2D(256, 256, 3, stride=1, pad=1)
            self.conv2_L2 = L.Convolution2D(256, 256, 3, stride=1, pad=1)
            self.conv3_L2 = L.Convolution2D(256, 256, 3, stride=1, pad=1)
            self.conv4_L2 = L.Convolution2D(256, joints, 1, stride=1, pad=0)

            self.Mconv1_stage2_L1 = L.Convolution2D(256+limbs+joints, 128, 7, stride=1, pad=3)
            self.Mconv2_stage2_L1 = L.Convolution2D(128, 128, 7, stride=1, pad=3)
            self.Mconv3_stage2_L1 = L.Convolution2D(128, 128, 7, stride=1, pad=3)
            self.Mconv4_stage2_L1 = L.Convolution2D(128, 128, 7, stride=1, pad=3)
            self.Mconv5_stage2_L1 = L.Convolution2D(128, 128, 7, stride=1, pad=3)
            self.Mconv6_stage2_L1 = L.Convolution2D(128, 128, 1, stride=1, pad=0)
            self.Mconv7_stage2_L1 = L.Convolution2D(128, limbs, 1, stride=1, pad=0)
            self.Mconv1_stage2_L2 = L.Convolution2D(256+limbs+joints, 128, 7, stride=1, pad=3)
            self.Mconv2_stage2_L2 = L.Convolution2D(128, 128, 7, stride=1, pad=3)
            self.Mconv3_stage2_L2 = L.Convolution2D(128, 128, 7, stride=1, pad=3)
            self.Mconv4_stage2_L2 = L.Convolution2D(128, 128, 7, stride=1, pad=3)
            self.Mconv5_stage2_L2 = L.Convolution2D(128, 128, 7, stride=1, pad=3)
            self.Mconv6_stage2_L2 = L.Convolution2D(128, 128, 1, stride=1, pad=0)
            self.Mconv7_stage2_L2 = L.Convolution2D(128, joints, 1, stride=1, pad=0)

    def __call__(self, x):
        pafs, heatmaps = [], []

        h2, h3, h4, h5 = self.res(x)

        h5n = self.newC5(h5)
        h4n = self.newC4(h4)
        h3n = self.newC3(h3)
        h2n = self.newC2(h2)

        hs = [h5n]
        h = self.upP5(h5n)
        h = h[:, :, :-1, :-1] + h4n
        hs.append(h)
        h = self.upP4(h)
        h = h + h3n
        hs.append(h)
        h = self.upP3(h)
        h = h + h2n
        hs.append(h)

        h5_, h4_, h3_, h2_ = hs

        h1 = F.relu(self.conv1_L1(h))
        h1 = F.relu(self.conv2_L1(h1))
        h1 = F.relu(self.conv3_L1(h1))
        h1 = F.relu(self.conv4_L1(h1))
        pafs.append(h1)
        h2 = F.relu(self.conv1_L2(h))
        h2 = F.relu(self.conv2_L2(h2))
        h2 = F.relu(self.conv3_L2(h2))
        h2 = F.relu(self.conv4_L2(h2))
        heatmaps.append(h2)

        h = F.concat((h1, h2, h), axis=1) # channel concat
        h1 = F.relu(self.Mconv1_stage2_L1(h)) # branch1
        h1 = F.relu(self.Mconv2_stage2_L1(h1))
        h1 = F.relu(self.Mconv3_stage2_L1(h1))
        h1 = F.relu(self.Mconv4_stage2_L1(h1))
        h1 = F.relu(self.Mconv5_stage2_L1(h1))
        h1 = F.relu(self.Mconv6_stage2_L1(h1))
        h1 = self.Mconv7_stage2_L1(h1)
        h2 = F.relu(self.Mconv1_stage2_L2(h)) # branch2
        h2 = F.relu(self.Mconv2_stage2_L2(h2))
        h2 = F.relu(self.Mconv3_stage2_L2(h2))
        h2 = F.relu(self.Mconv4_stage2_L2(h2))
        h2 = F.relu(self.Mconv5_stage2_L2(h2))
        h2 = F.relu(self.Mconv6_stage2_L2(h2))
        h2 = self.Mconv7_stage2_L2(h2)
        pafs.append(h1)
        heatmaps.append(h2)
        return pafs, heatmaps

if __name__ == '__main__':
    model = ResNet50()
    arr = np.random.rand(1, 3, 368, 368).astype('f')
    st = time.time()
    h1s, h2s = model(arr)
