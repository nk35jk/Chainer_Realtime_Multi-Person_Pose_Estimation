import time
import numpy as np
import chainer
from chainer import functions as F, links as L
from chainer.links import caffe


def copy_squeezenet_params(model):
    print('Copying params of pretrained model...')
    layer_names = [
        'conv1', 'fire2/squeeze1x1', 'fire2/expand1x1', 'fire2/expand3x3',
        'fire3/squeeze1x1', 'fire3/expand1x1', 'fire3/expand3x3',
        'fire4/squeeze1x1', 'fire4/expand1x1', 'fire4/expand3x3',
        'fire5/squeeze1x1', 'fire5/expand1x1', 'fire5/expand3x3',
        'fire6/squeeze1x1', 'fire6/expand1x1', 'fire6/expand3x3',
        'fire7/squeeze1x1', 'fire7/expand1x1', 'fire7/expand3x3',
        'fire8/squeeze1x1', 'fire8/expand1x1', 'fire8/expand3x3',
    ]
    pre_model = caffe.CaffeFunction('models/SqueezeNet_residual_top1_0.6038_top5_0.8250.caffemodel')
    for layer_name in layer_names:
        if len(layer_name.split('/')) > 1:
            exec('model.{}.{}.copyparams(pre_model["{}"])'.format(layer_name.split('/')[0], layer_name.split('/')[1], layer_name))
        else:
            exec('model.{}.copyparams(pre_model["{}"])'.format(layer_name, layer_name))
    print('Done.')


class Fire(chainer.Chain):
    def __init__(self, in_size, s1, e1, e3, res=False):
        super(Fire, self).__init__()
        with self.init_scope():
            self.squeeze1x1 = L.Convolution2D(in_size, s1, 1)
            self.expand1x1 = L.Convolution2D(s1, e1, 1)
            self.expand3x3 = L.Convolution2D(s1, e3, 3, pad=1)
        self.res = res

    def __call__(self, x):
        h = F.relu(self.squeeze1x1(x))
        h_1 = F.relu(self.expand1x1(h))
        h_3 = F.relu(self.expand3x3(h))
        h = F.concat([h_1, h_3], axis=1)
        if self.res:
            return x + h
        else:
            return h


class SqueezeNet(chainer.Chain):
    def __init__(self, res):
        super(SqueezeNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 96, 7, stride=2)
            self.fire2 = Fire(96, 16, 64, 64)
            self.fire3 = Fire(128, 16, 64, 64, res=res)
            self.fire4 = Fire(128, 32, 128, 128)
            self.fire5 = Fire(256, 32, 128, 128, res=res)
            self.fire6 = Fire(256, 48, 192, 192)
            self.fire7 = Fire(384, 48, 192, 192, res=res)
            self.fire8 = Fire(384, 64, 256, 256)

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 3, stride=2, pad=1)
        h = self.fire2(h)
        h = self.fire3(h)
        h = self.fire4(h)
        h = F.max_pooling_2d(h, 3, stride=2, pad=1)
        h = self.fire5(h)
        h = self.fire6(h)
        h = self.fire7(h)
        h = self.fire8(h)
        return h


class Student(chainer.Chain):
    """Student network based on SqueezeNet"""

    insize = 368
    downscale = 8
    pad = downscale

    def __init__(self, joints=19, limbs=38):
        super(Student, self).__init__()
        with self.init_scope():
            self.squeeze = SqueezeNet(res=True)
            self.fire9 = Fire(512, 64, 256, 256)
            self.fire10 = Fire(512, 64, 256, 256)
            self.fire11 = Fire(512, 64, 256, 256)
            self.fire12 = Fire(512, 64, 256, 256)
            self.conv13 = L.Convolution2D(512, limbs+joints, 1)
        self.joints = joints
        self.limbs = limbs

    def __call__(self, x):
        pafs, heatmaps = [], []

        h = self.squeeze(x)
        h = self.fire9(h)
        h = self.fire10(h)
        h = self.fire11(h)
        h = self.fire12(h)
        h = self.conv13(h)

        pafs.append(h[:, :self.limbs])
        heatmaps.append(h[:, -self.joints:])

        return pafs, heatmaps

if __name__ == '__main__':
    model = Student()
    arr = np.random.rand(1, 3, 368, 368).astype('f')
    st = time.time()
    y = model(arr)
