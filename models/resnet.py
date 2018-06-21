import chainer
import chainer.functions as F
import chainer.links as L
from models.resnet_chainer import ResNet50Layers, ResNet101Layers, ResNet152Layers


class ResNet(chainer.Chain):

    insize = 368
    downscale = pad = 8

    def __init__(self, joints=19, limbs=38, res=ResNet50Layers):
        super(ResNet, self).__init__()
        self.joints = joints
        self.limbs = limbs

        with self.init_scope():
            self.res = res()
            self.head1 = L.Convolution2D(2048, 512, 1)
            self.head2 = L.Convolution2D(512, 512, 3, 1, 1)
            self.head3 = L.Convolution2D(512, 512, 3, 1, 1)
            self.head4 = L.Convolution2D(512, 512, 3, 1, 1)
            self.head5 = L.Convolution2D(512, 512, 3, 1, 1)
            self.head6 = L.Convolution2D(512, 512, 3, 1, 1)
            self.head7 = L.Convolution2D(512, 512, 3, 1, 1)
            self.head8 = L.Convolution2D(512, 512, 3, 1, 1)
            self.head9 = L.Convolution2D(512, 512, 3, 1, 1)
            self.head10 = L.Convolution2D(512, limbs+joints, 1)

    def __call__(self, x):
        heatmaps = []
        pafs = []

        h = self.res(x, ['res5'])['res5']
        h = self.head1(h)
        h = self.head2(h)
        h = self.head3(h)
        h = self.head4(h)
        h = self.head5(h)
        h = self.head6(h)
        h = self.head7(h)
        h = self.head8(h)
        h = self.head9(h)
        h = self.head10(h)

        pafs = [h[:, :self.limbs]]
        heatmaps = [h[:, -self.joints:]]

        return pafs, heatmaps


class ResNet50(ResNet):

    def __init__(self, joints=19, limbs=38):
        super(ResNet50, self).__init__(joints, limbs, res=ResNet50Layers)


class ResNet101(ResNet):

    def __init__(self, joints=19, limbs=38):
        super(ResNet101, self).__init__(joints, limbs, res=ResNet101Layers)


class ResNet152(ResNet):

    def __init__(self, joints=19, limbs=38):
        super(ResNet152, self).__init__(joints, limbs, res=ResNet152Layers)


if __name__ == '__main__':
    chainer.config.enable_backprop = False
    chainer.config.train = False

    import time
    import numpy as np

    model = ResNet101()
    arr = np.random.randn(1, 3, model.insize, model.insize).astype('f')
    h1s, h2s = model(arr)
