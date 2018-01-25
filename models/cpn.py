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


class CPN(chainer.Chain):
    """Network like Cascaded Pyramid Network"""

    insize = 384
    downscale = 4
    pad = 32

    def __init__(self, joints=19, limbs=38):
        super(CPN, self).__init__()
        self.joints = joints
        self.limbs = limbs
        with self.init_scope():
            self.res = ResNet()
            self.C2lateral = L.Convolution2D(256, 256, 1)
            self.C3lateral = L.Convolution2D(512, 256, 1)
            self.C4lateral = L.Convolution2D(1024, 256, 1)
            self.C5lateral = L.Convolution2D(2048, 256, 1)

            self.convC3up = L.Convolution2D(256, 256, 1)
            self.convC4up = L.Convolution2D(256, 256, 1)
            self.convC5up = L.Convolution2D(256, 256, 1)

            self.convP2 = L.Convolution2D(256, 256, 3, pad=1)
            self.convP3 = L.Convolution2D(256, 256, 3, pad=1)
            self.convP4 = L.Convolution2D(256, 256, 3, pad=1)
            self.convP5 = L.Convolution2D(256, 256, 3, pad=1)

            self.conv1 = L.Convolution2D(1024, 512, 1, nobias=True)
            self.bn1 = L.BatchNormalization(512)
            self.conv2 = L.Convolution2D(512, 512, 3, pad=1, nobias=True)
            self.bn2 = L.BatchNormalization(512)
            self.conv3 = L.Convolution2D(512, 512, 3, pad=1, nobias=True)
            self.bn3 = L.BatchNormalization(512)
            self.conv4 = L.Convolution2D(512, 512, 3, pad=1, nobias=True)
            self.bn4 = L.BatchNormalization(512)
            self.conv5 = L.Convolution2D(512, 512, 3, pad=1, nobias=True)
            self.bn5 = L.BatchNormalization(512)
            self.conv6 = L.Convolution2D(512, 512, 3, pad=1, nobias=True)
            self.bn6 = L.BatchNormalization(512)
            self.conv7 = L.Convolution2D(512, 512, 3, pad=1, nobias=True)
            self.bn7 = L.BatchNormalization(512)
            self.conv8 = L.Convolution2D(512, limbs+joints, 1)

    def __call__(self, x):
        pafs, heatmaps = [], []

        c2, c3, c4, c5 = self.res(x)

        c5l = self.C5lateral(c5)
        c4l = self.C4lateral(c4)
        c3l = self.C3lateral(c3)
        c2l = self.C2lateral(c2)

        c5up = F.resize_images(c5l, (c5l.shape[2]*2, c5l.shape[3]*2))
        c5up = self.convC5up(c5up)
        c4sum = c4l + c5up
        c4up = F.resize_images(c4sum, (c4l.shape[2]*2, c4l.shape[3]*2))
        c4up = self.convC4up(c4up)
        c3sum = c3l + c4up
        c3up = F.resize_images(c3sum, (c3l.shape[2]*2, c3l.shape[3]*2))
        c3up = self.convC3up(c3up)
        c2sum = c2l + c3up

        p5 = self.convP5(c5l)
        p4 = self.convP4(c4sum)
        p3 = self.convP3(c3sum)
        p2 = self.convP2(c2sum)

        shape = c2l.shape[2:]
        p5r = F.resize_images(p5, shape)
        p4r = F.resize_images(p4, shape)
        p3r = F.resize_images(p3, shape)
        p2r = p2

        h = F.concat([p2r, p3r, p4r, p5r])

        h = F.relu(self.bn1(self.conv1(h)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.relu(self.bn4(self.conv4(h)))
        h = F.relu(self.bn5(self.conv5(h)))
        h = F.relu(self.bn6(self.conv6(h)))
        h = F.relu(self.bn7(self.conv7(h)))
        h = self.conv8(h)

        pafs.append(h[:, :self.limbs])
        heatmaps.append(h[:, -self.joints:])

        return pafs, heatmaps


if __name__ == '__main__':
    chainer.config.enable_backprop = False
    chainer.config.train = False

    model = CPN()
    arr = np.random.rand(1, 3, model.insize, model.insize).astype('f')
    h1s, h2s = model(arr)

    import chainer.computational_graph as c
    g = c.build_computational_graph(h1s)
    with open('graph.dot', 'w') as o:
        o.write(g.dump())
