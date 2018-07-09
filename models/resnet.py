import chainer
import chainer.functions as F
import chainer.links as L
from models.resnet_chainer import ResNet50Layers, ResNet101Layers, ResNet152Layers


class ResNet(chainer.Chain):

    insize = 368
    downscale = pad = 8

    def __init__(self, joints=19, limbs=38, res=ResNet50Layers, dilate=False):
        super(ResNet, self).__init__()
        self.joints = joints
        self.limbs = limbs

        with self.init_scope():
            self.res = res(dilate=dilate)
            self.head = L.Convolution2D(2048, limbs+joints, 1)

    def __call__(self, x):
        heatmaps = []
        pafs = []

        h = self.res(x, ['res5'])['res5']
        h = self.head(h)

        pafs = [h[:, :self.limbs]]
        heatmaps = [h[:, -self.joints:]]

        return pafs, heatmaps


class ResNet50(ResNet):

    def __init__(self, joints=19, limbs=38, dilate=False):
        super(ResNet50, self).__init__(joints, limbs, res=ResNet50Layers,
                                        dilate=dilate)


class ResNet101(ResNet):

    def __init__(self, joints=19, limbs=38, dilate=False):
        super(ResNet101, self).__init__(joints, limbs, res=ResNet101Layers,
                                        dilate=dilate)


class ResNet152(ResNet):

    def __init__(self, joints=19, limbs=38, dilate=False):
        super(ResNet152, self).__init__(joints, limbs, res=ResNet152Layers,
                                        dilate=dilate)


if __name__ == '__main__':
    chainer.config.enable_backprop = False
    chainer.config.train = False

    import time
    import numpy as np

    model = ResNet101(dilate=True)
    arr = np.random.randn(1, 3, model.insize, model.insize).astype('f')
    h1s, h2s = model(arr)
