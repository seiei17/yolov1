'''yolo  v1 architecture'''

import keras.backend as K
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Dropout
from keras.layers import LeakyReLU

from keras.models import Model
from keras.regularizers import l2
from keras.initializers import he_normal as initial


class yolov1:
    def __init__(self, S=7, B=2, w_decay=1e-4, leaky_alpha=0.1, dropout_rate=0.5):
        '''
        initial method of yolov1.
        :param S: # of grid;
        :param B: # of box per grid;
        :param w_decay: l2 weight decay;
        :param leaky_alpha: LeakyReLU alpha;
        :param dropout_rate: dropout rate;
        '''
        self.input_shape = (448, 448, 3,)
        self.num_classes = 20
        self.w_decay = w_decay
        self.alpha = leaky_alpha
        self.dropout_rate = dropout_rate
        self.S = S
        self.B = B
        self.output_size = self.S * self.S * \
                           (self.num_classes + 5 * self.B)

    def yolov1_net(self):
        '''
        yolo v1 networks.
        :return: net model
        '''
        net = {}
        net['input'] = Input(self.input_shape)
        net['conv1'] = Conv2D(64, 7, 2,
                              padding='same',
                              kernel_regularizer=l2(self.w_decay),
                              kernel_initializer=initial(),
                              )(net['input'])
        net['conv1'] = LeakyReLU(self.alpha)(net['conv1'])
        net['mp1'] = MaxPool2D((2, 2), strides=2, padding='same')(net['conv1'])
        net['conv2'] = Conv2D(192, 3,
                              padding='same',
                              kernel_regularizer=l2(self.w_decay),
                              kernel_initializer=initial(),
                              )(net['mp1'])
        net['conv2'] = LeakyReLU(self.alpha)(net['conv2'])
        net['mp2'] = MaxPool2D((2, 2), strides=2, padding='same')(net['conv2'])
        net['conv3'] = Conv2D(128, 1,
                              padding='same',
                              kernel_regularizer=l2(self.w_decay),
                              kernel_initializer=initial(),
                              )(net['mp2'])
        net['conv3'] = LeakyReLU(self.alpha)(net['conv3'])
        net['conv4'] = Conv2D(256, 3,
                              padding='same',
                              kernel_regularizer=l2(self.w_decay),
                              kernel_initializer=initial(),
                              )(net['conv3'])
        net['conv4'] = LeakyReLU(self.alpha)(net['conv4'])
        net['conv5'] = Conv2D(256, 1,
                              padding='same',
                              kernel_regularizer=l2(self.w_decay),
                              kernel_initializer=initial(),
                              )(net['conv4'])
        net['conv5'] = LeakyReLU(self.alpha)(net['conv5'])
        net['conv6'] = Conv2D(512, 3,
                              padding='same',
                              kernel_regularizer=l2(self.w_decay),
                              kernel_initializer=initial(),
                              )(net['conv5'])
        net['conv6'] = LeakyReLU(self.alpha)(net['conv6'])
        net['mp6'] = MaxPool2D((2, 2), strides=2, padding='same')(net['conv6'])
        mid = net['mp6']
        for i in range(4):
            mid = Conv2D(256, 1,
                         padding='same',
                         kernel_regularizer=l2(self.w_decay),
                         kernel_initializer=initial(),
                         )(mid)
            mid = LeakyReLU(self.alpha)(mid)
            mid = Conv2D(512, 3,
                         padding='same',
                         kernel_regularizer=l2(self.w_decay),
                         kernel_initializer=initial(),
                         )(mid)
            mid = LeakyReLU(self.alpha)(mid)
        net['conv714'] = mid
        net['conv15'] = Conv2D(512, 1,
                               padding='same',
                               kernel_regularizer=l2(self.w_decay),
                               kernel_initializer=initial(),
                               )(net['conv714'])
        net['conv15'] = LeakyReLU(self.alpha)(net['conv15'])
        net['conv16'] = Conv2D(1024, 3,
                               padding='same',
                               kernel_regularizer=l2(self.w_decay),
                               kernel_initializer=initial(),
                               )(net['conv15'])
        net['conv16'] = LeakyReLU(self.alpha)(net['conv16'])
        net['mp16'] = MaxPool2D((2, 2), strides=2, padding='same')(net['conv16'])
        mid = net['mp16']
        for i in range(2):
            mid = Conv2D(512, 1,
                         padding='same',
                         kernel_regularizer=l2(self.w_decay),
                         kernel_initializer=initial(),
                         )(mid)
            mid = LeakyReLU(self.alpha)(mid)
            mid = Conv2D(1024, 3,
                         padding='same',
                         kernel_regularizer=l2(self.w_decay),
                         kernel_initializer=initial(),
                         )(mid)
            mid = LeakyReLU(self.alpha)(mid)
        net['conv1720'] = mid
        net['conv21'] = Conv2D(1024, 3,
                               padding='same',
                               kernel_regularizer=l2(self.w_decay),
                               kernel_initializer=initial(),
                               )(net['conv1720'])
        net['conv21'] = LeakyReLU(self.alpha)(net['conv21'])
        net['conv22'] = Conv2D(2014, 3, strides=2,
                               padding='same',
                               kernel_regularizer=l2(self.w_decay),
                               kernel_initializer=initial(),
                               )(net['conv21'])
        net['conv22'] = LeakyReLU(self.alpha)(net['conv22'])
        net['conv23'] = Conv2D(1024, 3,
                               padding='same',
                               kernel_regularizer=l2(self.w_decay),
                               kernel_initializer=initial(),
                               )(net['conv22'])
        net['conv23'] = LeakyReLU(self.alpha)(net['conv23'])
        net['conv24'] = Conv2D(1024, 3,
                               padding='same',
                               kernel_regularizer=l2(self.w_decay),
                               kernel_initializer=initial(),
                               )(net['conv23'])
        net['conv24'] = LeakyReLU(self.alpha)(net['conv24'])
        net['flat'] = Flatten()(net['conv24'])
        net['fc25'] = Dense(4096)(net['flat'])
        net['fc25'] = LeakyReLU(self.alpha)(net['fc25'])
        net['dropout'] = Dropout(self.dropout_rate)(net['fc25'])
        net['fc26'] = Dense(self.output_size, activation='linear')(net['dropout'])
        net['output'] = net['fc26']
        model = Model(net['input'], net['output'])
        return model
