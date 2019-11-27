'''yolo v1 utils'''

import numpy as np
import tensorflow as tf


class YoloUtil(object):
    '''
    bounding box utils.

    # Arguments:
        num_classes: Number of classes except background;
        nms_threshold: threshold of nms;
    '''
    def __init__(self, num_classes, nms_threshold, top_k,
                 S=7, B=2,
                 obj_scale=1.0, noobj_scale=.5,
                 coord_scale=5.0, class_scale=1.0):
        self.num_classes = num_classes
        self._nms_threshold = nms_threshold
        self._top_k = top_k
        self.S = S
        self.B = B
        self.object_scale = obj_scale
        self.noobjec_scale = noobj_scale
        self.coord_scale = coord_scale
        self.class_scale = class_scale
        self.boundary1 =self.S * self.S* self.num_classes
        self.boundary2 =self.boundary1 +self.S * self.S * self.B


        self.boxes = tf.placeholder(dtype='float32', shape=(None, 4))
        self.scores = tf.placeholder(dtype='float32', shape=(None,))
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores,
                                                self._top_k,
                                                self._nms_threshold)
        self.sess = tf.Session()

    @property
    def nms_threshold(self):
        return self._nms_threshold
    @nms_threshold.setter
    def nms_threshold(self, value):
        self._nms_threshold = value
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores,
                                                self._top_k,
                                                self._nms_threshold)

    @property
    def top_k(self):
        return self._top_k
    @top_k.setter
    def top_k(self,value):
        self._top_k = value
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores,
                                                self._top_k,
                                                self._nms_threshold)

    def iou(self, box1, box2):
        '''
        compute iou between 2 boxes.
        :param box1: 5-d tensor, [None, S, S, B, 4] -> (x, y, w, h).
        :param box2: 5-d tensor, [None, S, S, B, 4] -> (x, y, w, h).
        :return:iou: 4-d tensor, [None, S, S, B].
        '''
        # Transferring (x, y, w, h) to (x1, y1, x2, y2).
        x11 = box1[:, :, :, :, 0] - box1[:, :, :, :, 2] / 2.0
        y11 = box1[:, :, :, :, 1] - box1[:, :, :, :, 3] / 2.0
        x12 = box1[:, :, :, :, 0] + box1[:, :, :, :, 2] / 2.0
        y12 = box1[:, :, :, :, 1] + box1[:, :, :, :, 3] / 2.0
        box1_new = tf.stack([x11, y11, x12, y12], axis=-1)
        x21 = box2[:, :, :, :, 0] - box2[:, :, :, :, 2] / 2.0
        y21 = box2[:, :, :, :, 1] - box2[:, :, :, :, 3] / 2.0
        x22 = box2[:, :, :, :, 0] + box2[:, :, :, :, 2] / 2.0
        y22 = box2[:, :, :, :, 1] + box2[:, :, :, :, 3] / 2.0
        box2_new = tf.stack([x21, y21, x22, y22], axis=-1)

        # Calculating 2 border points.
        leftupper = tf.maximum(box1_new[:, :, :, :, :2], box2_new[:, :, :, :, :2])
        rightdown = tf.minimum(box1_new[:, :, :, :, 2:], box2_new[:, :, :, :, 2:])

        # Calculating intersection.
        intersection = tf.maximum(0.0, leftupper - rightdown)
        inter_area = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]

        # Calculating area of box1 & box2.
        area1 = box1[:, :, :, :, 3] * box1[:, :, :, :, 2]
        area2 = box2[:, :, :, :, 3] * box2[:, :, :, :, 2]

        # Calculating union.
        union_area = tf.maximum(1e-10, area1 + area2 - inter_area)

        # Limiting the value into (0, 1).
        return tf.clip_by_value(inter_area / union_area, 0.0, 1.0)

    def loss_layer(self, predicts, labels):
        size = labels.shape[0]
        predict_classes = tf.reshape(predicts[:, :self.boundary1],
                                     (-1, self.S, self.S, self.num_classes))
        predict_conf = tf.reshape(predicts[:, self.boundary1: self.boundary2],
                                  (-1, self.S, self.S, self.B))
        predict_box = tf.reshape(predicts[:, self.boundary2:],
                                 (-1, self.S, self.S, self.B, 4))

        labels_response = tf.reshape(labels[:, :, :, 0],
                                    (-1, self.S, self.S, 1))
        labels_box = tf.reshape(labels[:, :, :, 1: 5],
                                (-1, self.S, self.S, 1, 4)) # shape is (, 7, 7, 1, 4)
        labels_box = tf.tile(labels_box, [1, 1, 1, self.B, 1]) # shape is (, 7, 7, 2, 4)
        labels_classes = labels[:, :, :, 5:]

        offset = np.transpose(np.reshape(np.array([np.arange(self.S)] * self.S * self.B),
                              (self.B, self.S, self.S)), (1, 2, 0))
        offset = tf.reshape(tf.constant(offset, dtype='float32'),
                            [1, self.S, self.S, self.B])
        offset_x = tf.tile(offset, [size, 1, 1, 1])
        offset_y = tf.transpose(offset_x, (0, 2, 1, 3))

        predict_box_tran = tf.stack([(predict_box[:, :, :, :, 0] + offset_x) / self.S,
                                     (predict_box[:, :, :, :, 1] + offset_y) / self.S,
                                     tf.square(predict_box[:, :, :, :, 2]),
                                     tf.square(predict_box[:, :, :, :, 3])], axis=-1)

        # shape is (, S, S, B)
        iou_preditct_truth = self.iou(predict_box_tran, labels_box)

        # Calculating I tensor (, S, S, B)
        # shape is (, S, S, 1)
        object_mask = tf.reduce_max(iou_preditct_truth, axis=-1, keep_dims=True)
        # shape is (, S, S, B)
        object_mask = tf.cast((iou_preditct_truth >= object_mask), tf.float32) * labels_response

        # Calculating no-I tensor (, S, S, B)
        noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask

        labels_box_trans = tf.stack([labels_box[:, :, :, :, 0] * self.S - offset_x,
                                     labels_box[:, :, :, :, 1] * self.S - offset_y,
                                     tf.sqrt(labels_box[:, :, :, :, 2]),
                                     tf.sqrt(labels_box[:, :, :, :, 3])], axis=-1)

        # loss of class
        class_differ = (predict_classes - labels_classes) * labels_response
        class_loss = tf.reduce_mean(tf.reduce_sum(tf.square(class_differ), axis=[1, 2, 3])) * self.class_scale

        # object loss
        object_differ = (predict_conf - iou_preditct_truth) * object_mask
        object_loss = tf.reduce_mean(tf.reduce_sum(tf.square(object_differ), axis=[1, 2, 3])) * self.object_scale

        # noobject loss
        noobject_differ = predict_conf * noobject_mask
        noobject_loss = tf.reduce_mean(tf.reduce_sum(tf.square(noobject_differ), axis=[1, 2, 3])) * self.noobjec_scale

        # coord loss
        # (, S, S, B) -> (, S, S, B, 4)
        coord_mask = tf.expand_dims(object_mask, 4)
        coord_differ = (predict_box - labels_box_trans) * coord_mask
        coord_loss = tf.reduce_mean(tf.reduce_sum(tf.square(coord_differ), axis=[1, 2, 3, 4])) * self.coord_scale

        tf.losses.add_loss(class_loss)
        tf.losses.add_loss(object_loss)
        tf.losses.add_loss(noobject_loss)
        tf.losses.add_loss(coord_loss)