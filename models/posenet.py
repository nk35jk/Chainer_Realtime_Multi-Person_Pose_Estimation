import chainer
import chainer.functions as F
import chainer.links as L
from chainer.links import caffe


class PoseNet(chainer.Chain):

    insize = 368
    downscale = pad = 8

    @classmethod
    def copy_vgg_params(cls, model):
        print('Copying params of pretrained model...')
        layer_names = [
            'conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1',
            'conv3_2', 'conv3_3', 'conv3_4', 'conv4_1', 'conv4_2',
        ]
        pre_model = caffe.CaffeFunction('models/VGG_ILSVRC_19_layers.caffemodel')
        for layer_name in layer_names:
            exec("model.%s.W.data = pre_model['%s'].W.data" % (layer_name, layer_name))
            exec("model.%s.b.data = pre_model['%s'].b.data" % (layer_name, layer_name))
        print('Done.')

    def __init__(self, joints=19, limbs=38, stages=6):
        super(PoseNet, self).__init__()
        self.stages = stages
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(3, 64, 3, stride=1, pad=1)
            self.conv1_2 = L.Convolution2D(64, 64, 3, stride=1, pad=1)
            self.conv2_1 = L.Convolution2D(64, 128, 3, stride=1, pad=1)
            self.conv2_2 = L.Convolution2D(128, 128, 3, stride=1, pad=1)
            self.conv3_1 = L.Convolution2D(128, 256, 3, stride=1, pad=1)
            self.conv3_2 = L.Convolution2D(256, 256, 3, stride=1, pad=1)
            self.conv3_3 = L.Convolution2D(256, 256, 3, stride=1, pad=1)
            self.conv3_4 = L.Convolution2D(256, 256, 3, stride=1, pad=1)
            self.conv4_1 = L.Convolution2D(256, 512, 3, stride=1, pad=1)
            self.conv4_2 = L.Convolution2D(512, 512, 3, stride=1, pad=1)
            self.conv4_3_CPM = L.Convolution2D(512, 256, 3, stride=1, pad=1)
            self.conv4_4_CPM = L.Convolution2D(256, 128, 3, stride=1, pad=1)

            # stage1
            self.conv5_1_CPM_L1 = L.Convolution2D(128, 128, 3, stride=1, pad=1)
            self.conv5_2_CPM_L1 = L.Convolution2D(128, 128, 3, stride=1, pad=1)
            self.conv5_3_CPM_L1 = L.Convolution2D(128, 128, 3, stride=1, pad=1)
            self.conv5_4_CPM_L1 = L.Convolution2D(128, 512, 1, stride=1, pad=0)
            self.conv5_5_CPM_L1 = L.Convolution2D(512, limbs, 1, stride=1, pad=0)

            self.conv5_1_CPM_L2 = L.Convolution2D(128, 128, 3, stride=1, pad=1)
            self.conv5_2_CPM_L2 = L.Convolution2D(128, 128, 3, stride=1, pad=1)
            self.conv5_3_CPM_L2 = L.Convolution2D(128, 128, 3, stride=1, pad=1)
            self.conv5_4_CPM_L2 = L.Convolution2D(128, 512, 1, stride=1, pad=0)
            self.conv5_5_CPM_L2 = L.Convolution2D(512, joints, 1, stride=1, pad=0)

        self.links = []
        for i in range(2, stages+1):
            self.add_stage(i, self.links, joints, limbs)
        for link in self.links:
            self.add_link(*link)

    def add_stage(self, stage, links, joints, limbs):
        in_channels = 128 + joints + limbs
        links += [('Mconv1_stage{}_L1'.format(stage), L.Convolution2D(in_channels, 128, 7, stride=1, pad=3))]
        links += [('Mconv2_stage{}_L1'.format(stage), L.Convolution2D(128, 128, 7, stride=1, pad=3))]
        links += [('Mconv3_stage{}_L1'.format(stage), L.Convolution2D(128, 128, 7, stride=1, pad=3))]
        links += [('Mconv4_stage{}_L1'.format(stage), L.Convolution2D(128, 128, 7, stride=1, pad=3))]
        links += [('Mconv5_stage{}_L1'.format(stage), L.Convolution2D(128, 128, 7, stride=1, pad=3))]
        links += [('Mconv6_stage{}_L1'.format(stage), L.Convolution2D(128, 128, 1, stride=1, pad=0))]
        links += [('Mconv7_stage{}_L1'.format(stage), L.Convolution2D(128, limbs, 1, stride=1, pad=0))]

        links += [('Mconv1_stage{}_L2'.format(stage), L.Convolution2D(in_channels, 128, 7, stride=1, pad=3))]
        links += [('Mconv2_stage{}_L2'.format(stage), L.Convolution2D(128, 128, 7, stride=1, pad=3))]
        links += [('Mconv3_stage{}_L2'.format(stage), L.Convolution2D(128, 128, 7, stride=1, pad=3))]
        links += [('Mconv4_stage{}_L2'.format(stage), L.Convolution2D(128, 128, 7, stride=1, pad=3))]
        links += [('Mconv5_stage{}_L2'.format(stage), L.Convolution2D(128, 128, 7, stride=1, pad=3))]
        links += [('Mconv6_stage{}_L2'.format(stage), L.Convolution2D(128, 128, 1, stride=1, pad=0))]
        links += [('Mconv7_stage{}_L2'.format(stage), L.Convolution2D(128, joints, 1, stride=1, pad=0))]

    def forward_stage(self, stage, h1, h2, h3, feature_map):
        h = F.concat((h1, h2, feature_map), axis=1)
        h1 = F.relu(self['Mconv1_stage{}_L1'.format(stage)](h))
        h1 = F.relu(self['Mconv2_stage{}_L1'.format(stage)](h1))
        h1 = F.relu(self['Mconv3_stage{}_L1'.format(stage)](h1))
        h1 = F.relu(self['Mconv4_stage{}_L1'.format(stage)](h1))
        h1 = F.relu(self['Mconv5_stage{}_L1'.format(stage)](h1))
        h1 = F.relu(self['Mconv6_stage{}_L1'.format(stage)](h1))
        h1 = self['Mconv7_stage{}_L1'.format(stage)](h1)

        h2 = F.relu(self['Mconv1_stage{}_L2'.format(stage)](h))
        h2 = F.relu(self['Mconv2_stage{}_L2'.format(stage)](h2))
        h2 = F.relu(self['Mconv3_stage{}_L2'.format(stage)](h2))
        h2 = F.relu(self['Mconv4_stage{}_L2'.format(stage)](h2))
        h2 = F.relu(self['Mconv5_stage{}_L2'.format(stage)](h2))
        h2 = F.relu(self['Mconv6_stage{}_L2'.format(stage)](h2))
        h2 = self['Mconv7_stage{}_L2'.format(stage)](h2)
        return h1, h2

    def __call__(self, x):
        heatmaps = []
        pafs = []

        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.relu(self.conv3_4(h))
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3_CPM(h))
        h = F.relu(self.conv4_4_CPM(h))
        feature_map = h

        # stage1
        h1 = F.relu(self.conv5_1_CPM_L1(feature_map))
        h1 = F.relu(self.conv5_2_CPM_L1(h1))
        h1 = F.relu(self.conv5_3_CPM_L1(h1))
        h1 = F.relu(self.conv5_4_CPM_L1(h1))
        h1 = self.conv5_5_CPM_L1(h1)
        pafs.append(h1)
        h2 = F.relu(self.conv5_1_CPM_L2(feature_map))
        h2 = F.relu(self.conv5_2_CPM_L2(h2))
        h2 = F.relu(self.conv5_3_CPM_L2(h2))
        h2 = F.relu(self.conv5_4_CPM_L2(h2))
        h2 = self.conv5_5_CPM_L2(h2)
        heatmaps.append(h2)

        # stage2~
        for i in range(2, self.stages+1):
            h1, h2 = self.forward_stage(i, h1, h2, None, feature_map)
            pafs.append(h1)
            heatmaps.append(h2)

        return pafs, heatmaps


if __name__ == '__main__':
    import time
    import numpy as np

    model = PoseNet()
    x_data = np.random.randn(1, 3, model.insize, model.insize).astype(np.float32)
    h1s, h2s = model(x_data)
