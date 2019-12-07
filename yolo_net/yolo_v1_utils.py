'''yolo v1 utils'''
import keras.backend as K
import cv2
import numpy as np
import tensorflow as tf


class YoloTrainUtils(object):
    '''
    bounding box utils.

    # Arguments:
        num_classes: Number of classes except background;
        nms_threshold: threshold of nms;
    '''

    def __init__(self, num_classes,
                 batch_size,
                 S=7, B=3,
                 obj_scale=1.0, noobj_scale=.5,
                 coord_scale=5.0, class_scale=1.0):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.S = S
        self.B = B
        self.object_scale = obj_scale
        self.noobjec_scale = noobj_scale
        self.coord_scale = coord_scale
        self.class_scale = class_scale
        self.boundary1 = self.S * self.S * self.num_classes
        self.boundary2 = self.boundary1 + self.S * self.S * self.B

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

    def loss_layer(self, labels, predicts):
        predict_classes = K.reshape(predicts[:, :self.boundary1],
                                     (-1, self.S, self.S, self.num_classes))
        predict_conf = K.reshape(predicts[:, self.boundary1: self.boundary2],
                                  (-1, self.S, self.S, self.B))
        predict_box = K.reshape(predicts[:, self.boundary2:],
                                 (-1, self.S, self.S, self.B, 4))

        labels_classes = K.reshape(labels[:, :self.boundary1],
                                  (-1, self.S, self.S, self.num_classes))
        labels_responses = K.reshape(labels[:, self.boundary1: self.boundary2],
                                     (-1, self.S, self.S, self.B))
        labels_response = labels_responses[:, :, :, 0]
        labels_box = K.reshape(labels[:, self.boundary2:],
                               (-1, self.S, self.S, self.B, 4))

        # predict_conf = K.concatenate([K.reshape(predicts[:, :, :, 0], (-1, self.S, self.S, 1)),
        #                               K.reshape(predicts[:, :, :, 5], (-1, self.S, self.S, 1))],
        #                              axis=3)
        # predict_box = K.reshape(K.concatenate([predicts[:, :, :, 1:5], predicts[:, :, :, 6:10]], axis=1),
        #                         (-1, self.S, self.S, self.B, 4))
        # predict_classes = predicts[:, :, :, 10:]
        #
        # labels_responses = K.concatenate([K.reshape(labels[:, :, :, 0], (-1, self.S, self.S, 1)),
        #                                   K.reshape(labels[:, :, :, 5], (-1, self.S, self.S, 1))],
        #                                  axis=3)
        # labels_response = labels_responses[:, :, :, 0]
        # labels_box = K.reshape(K.concatenate([labels[:, :, :, 1:5], labels[:, :, :, 6:10]], axis=1),
        #                        (-1, self.S, self.S, self.B, 4))
        # labels_classes = labels[:, :, :, 10:]

        offset = np.transpose(np.reshape(np.array([np.arange(self.S)] * self.S * self.B),
                                         (self.B, self.S, self.S)), (1, 2, 0))
        offset = K.reshape(tf.constant(offset, dtype='float32'),
                           [1, self.S, self.S, self.B])
        offset_x = tf.tile(offset, [self.batch_size, 1, 1, 1])
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
        object_mask = tf.cast((iou_preditct_truth >= object_mask), tf.float32) * labels_responses

        # Calculating no-I tensor (, S, S, B)
        noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask

        labels_box_trans = tf.stack([labels_box[:, :, :, :, 0] * self.S - offset_x,
                                     labels_box[:, :, :, :, 1] * self.S - offset_y,
                                     tf.sqrt(labels_box[:, :, :, :, 2]),
                                     tf.sqrt(labels_box[:, :, :, :, 3])], axis=-1)

        # loss of class
        class_differ = tf.reduce_sum(tf.square(predict_classes - labels_classes)) * labels_response
        class_loss = tf.reduce_mean(class_differ, axis=[1, 2]) * self.class_scale

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

        total_loss = class_loss + object_loss + noobject_loss + coord_loss
        return total_loss

class YoloDetector(object):
    def __init__(self, yolonet, YOLOv1_weight,
                 CLASSES, IMAGE_SIZE,
                 S=7, B=2,
                 THRESHOLD=0.45,
                 IOU_THRESHOLD=0.5):
        self.yolonet = yolonet
        self.YOLOv1_weight = YOLOv1_weight

        self.classes = CLASSES
        self.num_class = len(self.classes)
        self.image_size = IMAGE_SIZE
        self.S = S
        self.B =B
        self.threshold = THRESHOLD
        self.iou_threshold = IOU_THRESHOLD
        self.boundary1 = self.S * self.S * self.num_class
        self.boundary2 = self.boundary1 + \
                         self.S * self.S * self.B

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        print('Restoring weights from: ' + self.YOLOv1_weight)
        self.yolonet.load_weights(self.YOLOv1_weight, by_name=True)

    def draw_result(self, img, result):
        for i in range(len(result)):
            x = int(result[i][1])
            y = int(result[i][2])
            w = int(result[i][3] / 2)
            h = int(result[i][4] / 2)
            cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(img, (x - w, y - h - 20), (x + w, y - h), (125, 125, 125), -1)
            lineType = cv2.LINE_AA
            cv2.putText(
                img, result[i][0] + ' : %.2f' % result[i][5],
                (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1, lineType)

    def detect(self, img):
        img_h, img_w, _ = img.shape
        inputs = cv2.resize(img, (self.image_size, self.image_size))
        inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)
        inputs = (inputs / 255.0) * 2.0 - 1.0
        inputs = np.reshape(inputs, (1, self.image_size, self.image_size, 3))

        result = self.detect_from_cvmat(inputs)[0]

        for i in range(len(result)):
            result[i][1] *= (1.0 * img_w / self.image_size)
            result[i][2] *= (1.0 * img_h / self.image_size)
            result[i][3] *= (1.0 * img_w / self.image_size)
            result[i][4] *= (1.0 * img_h / self.image_size)

        return result

    def detect_from_cvmat(self, inputs):
        net_output = self.yolonet.predict(inputs, 1, 1)
        results = []
        for i in range(net_output.shape[0]):
            results.append(self.interpret_output(net_output[i]))

        return results

    def interpret_output(self, output):
        probs = np.zeros((self.S, self.S,
                          self.B, self.num_class))
        class_probs = np.reshape(
            output[0:self.boundary1],
            (self.S, self.S, self.num_class))
        scales = np.reshape(
            output[self.boundary1:self.boundary2],
            (self.S, self.S, self.B))
        boxes = np.reshape(
            output[self.boundary2:],
            (self.S, self.S, self.B, 4))
        offset = np.array(
            [np.arange(self.S)] * self.S * self.B)
        offset = np.transpose(
            np.reshape(
                offset,
                [self.B, self.S, self.S]),
            (1, 2, 0))

        boxes[:, :, :, 0] += offset
        boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
        boxes[:, :, :, :2] = 1.0 * boxes[:, :, :, 0:2] / self.S
        boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])

        boxes *= self.image_size

        for i in range(self.B):
            for j in range(self.num_class):
                probs[:, :, i, j] = np.multiply(class_probs[:, :, j], scales[:, :, i])

        filter_mat_probs = np.array(probs >= self.threshold, dtype='bool')
        filter_mat_boxes = np.nonzero(filter_mat_probs)
        boxes_filtered = boxes[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]
        probs_filtered = probs[filter_mat_probs]
        classes_num_filtered = np.argmax(filter_mat_probs, axis=3)[filter_mat_boxes[0],
                                                                   filter_mat_boxes[1],
                                                                   filter_mat_boxes[2]]

        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]

        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0:
                continue
            for j in range(i + 1, len(boxes_filtered)):
                if self.iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold:
                    probs_filtered[j] = 0.0

        filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]

        result = []
        for i in range(len(boxes_filtered)):
            result.append(
                [self.classes[classes_num_filtered[i]],
                 boxes_filtered[i][0],
                 boxes_filtered[i][1],
                 boxes_filtered[i][2],
                 boxes_filtered[i][3],
                 probs_filtered[i]])

        return result

    def iou(self, box1, box2):
        tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
             max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
             max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
        inter = 0 if tb < 0 or lr < 0 else tb * lr
        return inter / (box1[2] * box1[3] + box2[2] * box2[3] - inter)

    def camera_detector(self, cap, wait=10):
        ret, _ = cap.read()

        while ret:
            ret, frame = cap.read()
            result = self.detect(frame)
            self.draw_result(frame, result)
            cv2.imshow('Camera', frame)
            cv2.waitKey(wait)

            ret, frame = cap.read()

    def image_detector(self, imname, wait=0):
        image = cv2.imread(imname)

        result = self.detect(image)

        self.draw_result(image, result)
        cv2.imshow('Image', image)
        cv2.waitKey(wait)