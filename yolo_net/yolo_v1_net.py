'''yolo  v1 architecture'''

import keras.backend as K
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization

from keras.models import Model
from keras.regularizers import l2
from keras.initializers import he_normal as initial
from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint

from math import ceil
import os

from yolo_net.yolo_v1_utils import YoloTrainUtils


class yolov1:
    def __init__(self, S=7, B=3,
                 num_classes=20, BatchSize=128,
                 w_decay=1e-4,
                 leaky_alpha=0.1,
                 dropout_rate=0.4):
        '''
        initial method of yolov1.
        :param S: # of grid;
        :param B: # of box per grid;
        :param w_decay: l2 weight decay;
        :param leaky_alpha: LeakyReLU alpha;
        :param dropout_rate: dropout rate;
        '''
        self.input_shape = (448, 448, 3,)
        self.num_classes = num_classes
        self.BatchSize = BatchSize
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
        net['conv1'] = Conv2D(64, (7, 7),
                              strides=2,
                              padding='same',
                              kernel_regularizer=l2(self.w_decay),
                              kernel_initializer=initial(),
                              use_bias=False,
                              )(net['input'])
        net['bn1'] = BatchNormalization(axis=3)(net['conv1'])
        net['lr1'] = LeakyReLU(self.alpha)(net['bn1'])
        net['mp1'] = MaxPool2D((2, 2), strides=2, padding='same')(net['lr1'])

        net['conv2'] = Conv2D(192, (3, 3),
                              padding='same',
                              kernel_regularizer=l2(self.w_decay),
                              kernel_initializer=initial(),
                              use_bias=False,
                              )(net['mp1'])
        net['bn2'] = BatchNormalization(axis=3)(net['conv2'])
        net['lr2'] = LeakyReLU(self.alpha)(net['bn2'])
        net['mp2'] = MaxPool2D((2, 2), strides=2, padding='same')(net['lr2'])

        net['conv3'] = Conv2D(128, (1, 1),
                              padding='same',
                              kernel_regularizer=l2(self.w_decay),
                              kernel_initializer=initial(),
                              use_bias=False,
                              )(net['mp2'])
        net['bn3'] = BatchNormalization(axis=3)(net['conv3'])
        net['lr3'] = LeakyReLU(self.alpha)(net['bn3'])

        net['conv4'] = Conv2D(256, (3, 3),
                              padding='same',
                              kernel_regularizer=l2(self.w_decay),
                              kernel_initializer=initial(),
                              use_bias=False,
                              )(net['lr3'])
        net['bn4'] = BatchNormalization(axis=3)(net['conv4'])
        net['lr4'] = LeakyReLU(self.alpha)(net['bn4'])

        net['conv5'] = Conv2D(256, (1, 1),
                              padding='same',
                              kernel_regularizer=l2(self.w_decay),
                              kernel_initializer=initial(),
                              use_bias=False,
                              )(net['lr4'])
        net['bn5'] = BatchNormalization(axis=3)(net['conv5'])
        net['lr5'] = LeakyReLU(self.alpha)(net['bn5'])

        net['conv6'] = Conv2D(512, (3, 3),
                              padding='same',
                              kernel_regularizer=l2(self.w_decay),
                              kernel_initializer=initial(),
                              use_bias=False,
                              )(net['lr5'])
        net['bn6'] = BatchNormalization(axis=3)(net['conv6'])
        net['lr6'] = LeakyReLU(self.alpha)(net['bn6'])
        net['mp6'] = MaxPool2D((2, 2), strides=2, padding='same')(net['lr6'])
        mid = net['mp6']
        for i in range(4):
            mid = Conv2D(256, (1, 1),
                         padding='same',
                         kernel_regularizer=l2(self.w_decay),
                         kernel_initializer=initial(),
                         use_bias=False,
                         )(mid)
            mid = BatchNormalization(axis=3)(mid)
            mid = LeakyReLU(self.alpha)(mid)

            mid = Conv2D(512, (3, 3),
                         padding='same',
                         kernel_regularizer=l2(self.w_decay),
                         kernel_initializer=initial(),
                         use_bias=False,
                         )(mid)
            mid = BatchNormalization(axis=3)(mid)
            mid = LeakyReLU(self.alpha)(mid)
        net['conv714'] = mid
        net['conv15'] = Conv2D(512, (1, 1),
                               padding='same',
                               kernel_regularizer=l2(self.w_decay),
                               kernel_initializer=initial(),
                               use_bias=False,
                               )(net['conv714'])
        net['bn15'] = BatchNormalization(axis=3)(net['conv15'])
        net['lr15'] = LeakyReLU(self.alpha)(net['bn15'])

        net['conv16'] = Conv2D(1024, (3, 3),
                               padding='same',
                               kernel_regularizer=l2(self.w_decay),
                               kernel_initializer=initial(),
                               use_bias=False,
                               )(net['lr15'])
        net['bn16'] = BatchNormalization(axis=3)(net['conv16'])
        net['lr16'] = LeakyReLU(self.alpha)(net['bn16'])
        net['mp16'] = MaxPool2D((2, 2), strides=2, padding='same')(net['lr16'])

        mid = net['mp16']
        for i in range(2):
            mid = Conv2D(512, (1, 1),
                         padding='same',
                         kernel_regularizer=l2(self.w_decay),
                         kernel_initializer=initial(),
                         use_bias=False,
                         )(mid)
            mid = BatchNormalization(axis=3)(mid)
            mid = LeakyReLU(self.alpha)(mid)

            mid = Conv2D(1024, (3, 3),
                         padding='same',
                         kernel_regularizer=l2(self.w_decay),
                         kernel_initializer=initial(),
                         use_bias=False,
                         )(mid)
            mid = BatchNormalization(axis=3)(mid)
            mid = LeakyReLU(self.alpha)(mid)
        net['conv1720'] = mid
        net['conv21'] = Conv2D(1024, (3, 3),
                               padding='same',
                               kernel_regularizer=l2(self.w_decay),
                               kernel_initializer=initial(),
                               use_bias=False,
                               )(net['conv1720'])
        net['bn21'] = BatchNormalization(axis=3)(net['conv21'])
        net['lr21'] = LeakyReLU(self.alpha)(net['bn21'])

        net['conv22'] = Conv2D(1024, (3, 3), strides=2,
                               padding='same',
                               kernel_regularizer=l2(self.w_decay),
                               kernel_initializer=initial(),
                               use_bias=False,
                               )(net['lr21'])
        net['bn22'] = BatchNormalization(axis=3)(net['conv22'])
        net['lr22'] = LeakyReLU(self.alpha)(net['bn22'])

        net['conv23'] = Conv2D(1024, (3, 3),
                               padding='same',
                               kernel_regularizer=l2(self.w_decay),
                               kernel_initializer=initial(),
                               use_bias=False,
                               )(net['lr22'])
        net['bn23'] = BatchNormalization(axis=3)(net['conv23'])
        net['lr23'] = LeakyReLU(self.alpha)(net['bn23'])

        net['conv24'] = Conv2D(1024, (3, 3),
                               padding='same',
                               kernel_regularizer=l2(self.w_decay),
                               kernel_initializer=initial(),
                               use_bias=False,
                               )(net['lr23'])
        net['bn24'] = BatchNormalization(axis=3)(net['conv24'])
        net['lr24'] = LeakyReLU(self.alpha)(net['bn24'])

        net['local_conv'] = Conv2D(256, (3, 3),
                              padding='same',
                              kernel_regularizer=l2(self.w_decay),
                              kernel_initializer=initial(),
                              )(net['lr24'])
        net['local_lr'] = LeakyReLU(self.alpha)(net['local_conv'])

        net['reduce_conv'] = Conv2D(128, (1, 1),
                                    padding='same',
                                    kernel_regularizer=l2(self.w_decay),
                                    kernel_initializer=initial(),
                                    )(net['local_lr'])
        net['reduce_lr'] = LeakyReLU(self.alpha)(net['reduce_conv'])

        net['flat'] = Flatten()(net['reduce_lr'])
        net['dropout'] = Dropout(self.dropout_rate)(net['flat'])
        net['fc26'] = Dense(self.output_size, activation='linear')(net['dropout'])
        net['output'] = net['fc26']

        model = Model(net['input'], net['output'])
        # model.summary()
        return model

    def train(self, pascal, epochs, lr):
        checkpoint = './history/YOLOv1_weight.h5'
        yoloutils = YoloTrainUtils(self.num_classes, B=self.B, batch_size=self.BatchSize)
        model = self.yolov1_net()
        loss = yoloutils.loss_layer
        opt = Adam(lr)
        callback = ModelCheckpoint(checkpoint,
                                   monitor='accuracy',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=True)

        length = pascal.len
        step = ceil(length / self.BatchSize)
        model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

        if os.path.isfile(checkpoint):
            print('---\nload weights from {}\n----'.format(checkpoint))
            model.load_weights(checkpoint, by_name=True)


        model.fit_generator(pascal.generator(),
                            step,
                            epochs=epochs,
                            verbose=1,
                            callbacks=[callback],
                            )