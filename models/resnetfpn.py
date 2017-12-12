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


class ResNetFPN(chainer.Chain):
    """resnet50-FPN"""

    insize = 384
    downscale = 4

    def __init__(self, joints=19, limbs=38, compute_mask=False):
        super(ResNetFPN, self).__init__()
        with self.init_scope():
            self.res = ResNet()
            self.C5lateral = L.Convolution2D(2048, 256, 1, stride=1, pad=0)
            self.C4lateral = L.Convolution2D(1024, 256, 1, stride=1, pad=0)
            self.C3lateral = L.Convolution2D(512, 256, 1, stride=1, pad=0)
            self.C2lateral = L.Convolution2D(256, 256, 1, stride=1, pad=0)

            self.convP4 = L.Convolution2D(256, 256, 3, stride=1, pad=1)
            self.convP3 = L.Convolution2D(256, 256, 3, stride=1, pad=1)
            self.convP2 = L.Convolution2D(256, 256, 3, stride=1, pad=1)

            self.conv1_L1 = L.Convolution2D(256, 128, 3, stride=1, pad=1)
            self.conv2_L1 = L.Convolution2D(128, 128, 3, stride=1, pad=1)
            self.conv3_L1 = L.Convolution2D(128, 128, 3, stride=1, pad=1)
            self.conv4_L1 = L.Convolution2D(128, 128, 1, stride=1, pad=0)
            self.conv5_L1 = L.Convolution2D(128, limbs, 1, stride=1, pad=0)
            self.conv1_L2 = L.Convolution2D(256, 128, 3, stride=1, pad=1)
            self.conv2_L2 = L.Convolution2D(128, 128, 3, stride=1, pad=1)
            self.conv3_L2 = L.Convolution2D(128, 128, 3, stride=1, pad=1)
            self.conv4_L2 = L.Convolution2D(128, 128, 1, stride=1, pad=0)
            self.conv5_L2 = L.Convolution2D(128, joints, 1, stride=1, pad=0)

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

        c2, c3, c4, c5 = self.res(x)

        c5l = self.C5lateral(c5)
        c4l = self.C4lateral(c4)
        c3l = self.C3lateral(c3)
        c2l = self.C2lateral(c2)

        p5 = c5l

        h = F.resize_images(p5, (p5.shape[2]*2, p5.shape[3]*2))
        h = h[:, :, :c4l.shape[2], :c4l.shape[3]] + c4l
        p4 = self.convP4(h)

        h = F.resize_images(p4, (p4.shape[2]*2, p4.shape[3]*2))
        h = h[:, :, :c3l.shape[2], :c3l.shape[3]] + c3l
        p3 = self.convP3(h)

        h = F.resize_images(p3, (p3.shape[2]*2, p3.shape[3]*2))
        h = h[:, :, :c2l.shape[2], :c2l.shape[3]] + c2l
        p2 = self.convP2(h)

        h1 = F.relu(self.conv1_L1(p2))
        h1 = F.relu(self.conv2_L1(h1))
        h1 = F.relu(self.conv3_L1(h1))
        h1 = F.relu(self.conv4_L1(h1))
        h1 = self.conv5_L1(h1)
        pafs.append(h1)
        h2 = F.relu(self.conv1_L2(p2))
        h2 = F.relu(self.conv2_L2(h2))
        h2 = F.relu(self.conv3_L2(h2))
        h2 = F.relu(self.conv4_L2(h2))
        h2 = self.conv5_L2(h2)
        heatmaps.append(h2)

        h = F.concat((h1, h2, p2), axis=1) # channel concat
        h1 = F.relu(self.Mconv1_stage2_L1(h)) # branch1
        h1 = F.relu(self.Mconv2_stage2_L1(h1))
        h1 = F.relu(self.Mconv3_stage2_L1(h1))
        h1 = F.relu(self.Mconv4_stage2_L1(h1))
        h1 = F.relu(self.Mconv5_stage2_L1(h1))
        h1 = F.relu(self.Mconv6_stage2_L1(h1))
        h1 = self.Mconv7_stage2_L1(h1)
        pafs.append(h1)
        h2 = F.relu(self.Mconv1_stage2_L2(h)) # branch2
        h2 = F.relu(self.Mconv2_stage2_L2(h2))
        h2 = F.relu(self.Mconv3_stage2_L2(h2))
        h2 = F.relu(self.Mconv4_stage2_L2(h2))
        h2 = F.relu(self.Mconv5_stage2_L2(h2))
        h2 = F.relu(self.Mconv6_stage2_L2(h2))
        h2 = self.Mconv7_stage2_L2(h2)
        heatmaps.append(h2)
        return pafs, heatmaps


if __name__ == '__main__':
    model = ResNetFPN()
    arr = np.random.rand(1, 3, 384, 384).astype('f')
    st = time.time()
    h1s, h2s = model(arr)

    import chainer.computational_graph as c
    g = c.build_computational_graph(h1s)
    with open('graph.dot', 'w') as o:
        o.write(g.dump())
