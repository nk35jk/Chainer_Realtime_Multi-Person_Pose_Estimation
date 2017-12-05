import time
import numpy as np
import chainer
from chainer import functions as F, links as L
from chainer.links import caffe


def copy_squeezenet_params(model):
    print('Copying params of pretrained model...')
    layer_names = [
        "conv1", "fire2/squeeze1x1", "fire2/expand1x1", "fire2/expand3x3",
        "fire3/squeeze1x1", "fire3/expand1x1", "fire3/expand3x3",
        "fire4/squeeze1x1", "fire4/expand1x1", "fire4/expand3x3",
        "fire5/squeeze1x1", "fire5/expand1x1", "fire5/expand3x3",
        "fire6/squeeze1x1", "fire6/expand1x1", "fire6/expand3x3",
        "fire7/squeeze1x1", "fire7/expand1x1", "fire7/expand3x3",
        "fire8/squeeze1x1", "fire8/expand1x1", "fire8/expand3x3",
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
        super(Fire, self).__init__(
            squeeze1x1=L.Convolution2D(in_size, s1, 1),
            expand1x1=L.Convolution2D(s1, e1, 1),
            expand3x3=L.Convolution2D(s1, e3, 3, pad=1),
        )
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
        super(SqueezeNet, self).__init__(
            conv1=L.Convolution2D(3, 96, 7, stride=2),
            fire2=Fire(96, 16, 64, 64),
            fire3=Fire(128, 16, 64, 64, res=res),
            fire4=Fire(128, 32, 128, 128),
            fire5=Fire(256, 32, 128, 128, res=res),
            fire6=Fire(256, 48, 192, 192),
            fire7=Fire(384, 48, 192, 192, res=res),
            fire8=Fire(384, 64, 256, 256),
        )

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


class StageN(chainer.Chain):
    def __init__(self, joints, limbs, stuffs):
        super(StageN, self).__init__(
            conv1_1=L.Convolution2D(128+joints+limbs, 128, 3, stride=1, pad=1),
            conv1_2=L.DilatedConvolution2D(128, 128, 3, stride=1, pad=2, dilate=2),
            conv1_3=L.DilatedConvolution2D(128, 128, 3, stride=1, pad=4, dilate=4),
            conv1_4=L.DilatedConvolution2D(128, 128, 3, stride=1, pad=8, dilate=8),

            conv1_L1=L.Convolution2D(128, 128, 3, stride=1, pad=1),
            conv2_L1=L.Convolution2D(128, 128, 3, stride=1, pad=1),
            conv3_L1=L.Convolution2D(128, 128, 3, stride=1, pad=1),
            conv4_L1=L.Convolution2D(128, 128, 3, stride=1, pad=1),
            conv5_L1=L.Convolution2D(128, 128, 1, stride=1, pad=0),
            conv6_L1=L.Convolution2D(128, limbs, 1, stride=1, pad=0),

            conv1_L2=L.Convolution2D(128, 128, 3, stride=1, pad=1),
            conv2_L2=L.Convolution2D(128, 128, 3, stride=1, pad=1),
            conv3_L2=L.Convolution2D(128, 128, 3, stride=1, pad=1),
            conv4_L2=L.Convolution2D(128, 128, 3, stride=1, pad=1),
            conv5_L2=L.Convolution2D(128, 128, 1, stride=1, pad=0),
            conv6_L2=L.Convolution2D(128, joints, 1, stride=1, pad=0),

            conv1_L3=L.Convolution2D(128, 128, 3, stride=1, pad=1),
            conv2_L3=L.Convolution2D(128, 128, 3, stride=1, pad=1),
            conv3_L3=L.Convolution2D(128, 128, 3, stride=1, pad=1),
            conv4_L3=L.Convolution2D(128, 128, 3, stride=1, pad=1),
            conv5_L3=L.Convolution2D(128, 128, 1, stride=1, pad=0),
            conv6_L3=L.Convolution2D(128, stuffs, 1, stride=1, pad=0),
        )

    def __call__(self, h1, h2, h3, feature_map):
        h = F.concat((h1, h2, feature_map), axis=1)
        h = F.relu(self.conv1_1(h))
        h = F.relu(self.conv1_2(h))
        h = F.relu(self.conv1_3(h))
        h = F.relu(self.conv1_4(h))

        h1 = F.relu(self.conv1_L1(h))
        h1 = F.relu(self.conv2_L1(h1))
        h1 = F.relu(self.conv3_L1(h1))
        h1 = F.relu(self.conv4_L1(h1))
        h1 = F.relu(self.conv5_L1(h1))
        h1 = self.conv6_L1(h1)

        h2 = F.relu(self.conv1_L2(h))
        h2 = F.relu(self.conv2_L2(h2))
        h2 = F.relu(self.conv3_L2(h2))
        h2 = F.relu(self.conv4_L2(h2))
        h2 = F.relu(self.conv5_L2(h2))
        h2 = self.conv6_L2(h2)

        h3 = F.relu(self.conv1_L3(h))
        h3 = F.relu(self.conv2_L3(h3))
        h3 = F.relu(self.conv3_L3(h3))
        h3 = F.relu(self.conv4_L3(h3))
        h3 = F.relu(self.conv5_L3(h3))
        h3 = self.conv6_L3(h3)
        return h1, h2, h3


class NN1(chainer.Chain):
    """SqueezeNet + 6 Stages including Dilated Conv"""

    insize = 368
    downscale = 8

    def __init__(self, joints=19, limbs=38, stuffs=182, stage=6, compute_mask=False):
        super(NN1, self).__init__(
            squeeze=SqueezeNet(res=True),
            conv1_1=L.Convolution2D(512, 256, 3, stride=1, pad=1),
            conv1_2=L.Convolution2D(256, 128, 3, stride=1, pad=1),
            bn1=L.BatchNormalization(128),

            # stage1
            conv1_L1=L.Convolution2D(128, 128, 3, stride=1, pad=1),
            conv2_L1=L.Convolution2D(128, 128, 3, stride=1, pad=1),
            conv3_L1=L.Convolution2D(128, 128, 3, stride=1, pad=1),
            conv4_L1=L.Convolution2D(128, 512, 1, stride=1, pad=0),
            conv5_L1=L.Convolution2D(512, limbs, 1, stride=1, pad=0),

            conv1_L2=L.Convolution2D(128, 128, 3, stride=1, pad=1),
            conv2_L2=L.Convolution2D(128, 128, 3, stride=1, pad=1),
            conv3_L2=L.Convolution2D(128, 128, 3, stride=1, pad=1),
            conv4_L2=L.Convolution2D(128, 512, 1, stride=1, pad=0),
            conv5_L2=L.Convolution2D(512, joints, 1, stride=1, pad=0),

            conv1_L3=L.Convolution2D(128, 128, 3, stride=1, pad=1),
            conv2_L3=L.Convolution2D(128, 128, 3, stride=1, pad=1),
            conv3_L3=L.Convolution2D(128, 128, 3, stride=1, pad=1),
            conv4_L3=L.Convolution2D(128, 512, 1, stride=1, pad=0),
            conv5_L3=L.Convolution2D(512, stuffs, 1, stride=1, pad=0),
        )
        links = [('stage{}'.format(i), StageN(joints, limbs, stuffs)) for i in range(2, stage+1)]
        [self.add_link(*l) for l in links]
        self.stagen = links

    def __call__(self, x):
        y1s, y2s, y3s = [], [], []

        st1 = time.time()
        h = self.squeeze(x)
        h = F.relu(self.conv1_1(h))
        h = F.relu(self.conv1_2(h))
        h = self.bn1(h)
        feature_map = h
        # print('squeeze: {:.4f}s'.format(time.time() - st1))
        st = time.time()

        # stage1
        h1 = F.relu(self.conv1_L1(feature_map))
        h1 = F.relu(self.conv2_L1(h1))
        h1 = F.relu(self.conv3_L1(h1))
        h1 = F.relu(self.conv4_L1(h1))
        h1 = self.conv5_L1(h1)
        y1s.append(h1)
        h2 = F.relu(self.conv1_L2(feature_map))
        h2 = F.relu(self.conv2_L2(h2))
        h2 = F.relu(self.conv3_L2(h2))
        h2 = F.relu(self.conv4_L2(h2))
        h2 = self.conv5_L2(h2)
        y2s.append(h2)
        if self.compute_mask:
            h3 = F.relu(self.conv1_L3(feature_map))
            h3 = F.relu(self.conv2_L3(h3))
            h3 = F.relu(self.conv3_L3(h3))
            h3 = F.relu(self.conv4_L3(h3))
            h3 = self.conv5_L3(h3)
            y3s.append(h3)
        # print('stage1: {:.4f}s'.format(time.time() - st))

        # stage2~
        for name, stage in self.stagen:
            st = time.time()
            h1, h2, h3 = stage(h1, h2, h3, feature_map)
            y1s.append(h1)
            y2s.append(h2)
            if self.compute_mask:
                y3s.append(h3)
            # print('{}: {:.4f}s'.format(name, time.time() - st))
        # print('forward: {:.4f}s'.format(time.time() - st1))
        return y1s, y2s, y3s

if __name__ == '__main__':
    model = NN1()
    arr = np.random.rand(1, 3, 368, 368).astype('f')
    st = time.time()
    y = model(arr)
