import chainer
import chainer.functions as F
import chainer.links as L


class Student(chainer.Chain):
    """small network"""

    insize = 368
    downscale = pad = 8

    def __init__(self, joints=19, limbs=38):
        super(Student, self).__init__()
        self.joints = joints
        self.limbs = limbs
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(3, 32, 3, 1, 1)
            self.conv1_2 = L.Convolution2D(32, 32, 3, 1, 1)
            self.conv2_1 = L.Convolution2D(32, 64, 3, 1, 1)
            self.conv2_2 = L.Convolution2D(64, 64, 3, 1, 1)
            self.conv3_1 = L.Convolution2D(64, 128, 3, 1, 1)
            self.conv3_2 = L.Convolution2D(128, 128, 3, 1, 1)
            self.conv3_3 = L.Convolution2D(128, 128, 3, 1, 1)
            self.conv3_4 = L.Convolution2D(128, 128, 3, 1, 1)
            self.conv4_1 = L.Convolution2D(128, 128, 3, 1, 1)
            self.conv4_2 = L.Convolution2D(128, 128, 3, 1, 1)
            self.conv4_3 = L.Convolution2D(128, 128, 3, 1, 1)
            self.conv4_4 = L.Convolution2D(128, 128, 3, 1, 1)
            self.conv5_1 = L.Convolution2D(128, 128, 7, 1, 3)
            self.conv5_2 = L.Convolution2D(128, 128, 7, 1, 3)
            self.conv5_3 = L.Convolution2D(128, 128, 7, 1, 3)
            self.conv5_4 = L.Convolution2D(128, 128, 7, 1, 3)
            self.conv5_5 = L.Convolution2D(128, 128, 7, 1, 3)
            self.conv5_6 = L.Convolution2D(128, 128, 1, 1)
            self.conv5_7 = L.Convolution2D(128, limbs+joints, 1)

    def __call__(self, x):
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.relu(self.conv3_4(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = F.relu(self.conv4_4(h))

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.relu(self.conv5_4(h))
        h = F.relu(self.conv5_5(h))
        h = F.relu(self.conv5_6(h))
        h = self.conv5_7(h)

        pafs = [h[:, :self.limbs]]
        heatmaps = [h[:, -self.joints:]]

        return pafs, heatmaps


if __name__ == '__main__':
    import time
    import numpy as np

    chainer.config.enable_backprop = False
    chainer.config.train = False

    model = Student()
    arr = np.random.randn(1, 3, model.insize, model.insize).astype('f')
    h1s, h2s = model(arr)
