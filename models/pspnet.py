import time
import numpy as np
import chainer
from chainer import functions as F, links as L
from chainer import initializers
from chainer.links import caffe


class BottleNeckA(chainer.Chain):
    def __init__(self, in_size, ch, out_size, stride=1, dilation=1):
        super(BottleNeckA, self).__init__()
        initialW = initializers.HeNormal()

        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_size, ch, 1, stride, 0, initialW=initialW, nobias=True)
            self.bn1 = L.BatchNormalization(ch)
            if dilation == 1:
                self.conv2 = L.Convolution2D(
                    ch, ch, 3, 1, 1, initialW=initialW, nobias=True)
            else:
                self.conv2 = L.DilatedConvolution2D(ch, ch, 3, 1, pad=dilation,
                    dilate=dilation, initialW=initialW, nobias=True)
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
    def __init__(self, in_size, ch, dilation=1):
        super(BottleNeckB, self).__init__()
        initialW = initializers.HeNormal()

        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_size, ch, 1, 1, 0, initialW=initialW, nobias=True)
            self.bn1 = L.BatchNormalization(ch)
            if dilation == 1:
                self.conv2 = L.Convolution2D(
                    ch, ch, 3, 1, 1, initialW=initialW, nobias=True)
            else:
                self.conv2 = L.DilatedConvolution2D(ch, ch, 3, 1, pad=dilation,
                    dilate=dilation, initialW=initialW, nobias=True)
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
    def __init__(self, layer, in_size, ch, out_size, stride=2, dilation=1):
        super(Block, self).__init__()
        self.add_link('a', BottleNeckA(in_size, ch, out_size, stride, dilation))
        for i in range(1, layer):
            self.add_link('b{}'.format(i), BottleNeckB(out_size, ch, dilation))
        self.layer = layer

    def __call__(self, x):
        h = self.a(x)
        for i in range(1, self.layer):
            h = self['b{}'.format(i)](h)
        return h


class ResNet(chainer.Chain):
    def __init__(self):
        super(ResNet, self).__init__()
        initialW = initializers.HeNormal()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                3, 64, 7, 2, 3, initialW=initializers.HeNormal(), nobias=True)
            self.bn1 = L.BatchNormalization(64)

            self.res2 = Block(3, 64, 64, 256, stride=1)
            self.res3 = Block(4, 256, 128, 512, stride=2)

    def __call__(self, x):
        hs = []
        h = self.bn1(self.conv1(x))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = self.res2(h)
        hs.append(h)
        h = self.res3(h)
        hs.append(h)
        return hs


class PSPNet(chainer.Chain):
    """Network like Pyramid Scene Pursing Network"""

    insize = 480
    downscale = pad = 8

    def __init__(self, joints=19, limbs=38, compute_mask=False):
        super(PSPNet, self).__init__()
        self.joints = joints
        self.limbs = limbs
        with self.init_scope():
            self.res = ResNet()

            self.res4 = Block(6, 512, 256, 1024, stride=1, dilation=2)
            self.res5 = Block(3, 1024, 512, 2048, stride=1, dilation=4)

            self.conv5_3 = L.Convolution2D(2048, 512, 1, nobias=True)
            self.bn5_3 = L.BatchNormalization(512)

            self.conv5_4 = L.Convolution2D(2560, 512, 1, nobias=True)
            self.bn5_4 = L.BatchNormalization(512)
            self.conv6 = L.Convolution2D(512, limbs+joints, 1)

    def __call__(self, x):
        pafs, heatmaps = [], []

        c2, c3 = self.res(x)

        c4 = self.res4(c3)
        c5 = self.res5(c4)

        shape = c5.shape[2:]
        pool1 = F.average_pooling_2d(c5, ksize=shape)
        pool1 = F.relu(self.bn5_3(self.conv5_3(pool1)))
        pool1 = F.resize_images(pool1, shape)

        h = F.concat([c5, pool1])

        h = F.relu(self.bn5_4(self.conv5_4(h)))
        h = self.conv6(h)

        pafs.append(h[:, :self.limbs])
        heatmaps.append(h[:, -self.joints:])

        return pafs, heatmaps


if __name__ == '__main__':
    model = PSPNet()
    arr = np.random.rand(1, 3, model.insize, model.insize).astype('f')
    h1s, h2s = model(arr)

    import chainer.computational_graph as c
    g = c.build_computational_graph(h1s)
    with open('graph.dot', 'w') as o:
        o.write(g.dump())
