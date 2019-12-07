import os
import cv2
import pickle
import copy
import numpy as np
import xml.etree.ElementTree as ET

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']


class pascal_voc(object):
    def __init__(self, PASCAL_PATH, TrainOrTest, S=7, B=2, FLIPPED=True, REBUILD=False):
        self.pascal_path = PASCAL_PATH
        self.devkit_path = os.path.join(self.pascal_path, 'VOCdevkit')
        self.data_path = os.path.join(self.devkit_path, 'VOC2007')
        self.cache_path = os.path.join(self.pascal_path, 'cache')
        self.img_size = 448
        self.cls = CLASSES
        self.s = S
        self.b = B
        self.one_vec = self.s * self.s * (self.b * 5 + 20)
        self.cls_to_idx = dict(zip(self.cls, range(len(self.cls))))
        self.flipped = FLIPPED
        self.rebuild = REBUILD
        self.TrainOrTest = TrainOrTest
        self.cursor = 0
        self.epoch = 1
        self.gt_labels = None
        self.len = 0
        self.load_labels()

    def load_labels(self):
        cache_file = os.path.join(self.cache_path,
                                  'pascal_' +
                                  self.TrainOrTest +
                                  '_gt_labels.pkl')

        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        if self.TrainOrTest == 'train':
            txtname = os.path.join(self.data_path, 'ImageSets', 'Main', 'trainval.txt')
        else:
            txtname = os.path.join(self.data_path, 'ImageSets', 'Main', 'val.txt')
        self.image_index = '000033'

        gt_labels = []
        index = self.image_index
        label, num = self.load_pascal_annotation(index)
        imgname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
        gt_labels.append({'imgname': imgname,
                          'label': label,
                          'flipped': False})
        print('Saving gt_labels to: ' + cache_file)
        with open(cache_file, 'wb') as file:
            pickle.dump(gt_labels, file)

    def load_pascal_annotation(self, index):
        '''
        Load image and bounding boxes info from XML.
        :param index:
        :return:
        '''
        img_name = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
        img = cv2.imread(img_name)
        h_ratio = 1.0 * self.img_size / img.shape[0]
        w_ratio = 1.0 * self.img_size / img.shape[1]
        # img = cv2.resize(img, [self.img_size, self.img_size])

        label = np.zeros((self.s, self.s, self.b * 5 + 20))
        xml_name = os.path.join(self.data_path, 'Annotations', index + '.xml')
        tree = ET.parse(xml_name)
        objs = tree.findall('object')

        for obj in objs:
            bbox = obj.find('bndbox')

            x1 = max(min((float(bbox.find('xmin').text) - 1) * w_ratio, self.img_size - 1), 0)
            y1 = max(min((float(bbox.find('ymin').text) - 1) * h_ratio, self.img_size - 1), 0)
            x2 = max(min((float(bbox.find('xmax').text) - 1) * w_ratio, self.img_size - 1), 0)
            y2 = max(min((float(bbox.find('ymax').text) - 1) * h_ratio, self.img_size - 1), 0)

            cls_idx = self.cls_to_idx[obj.find('name').text.lower().strip()]
            box = [(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1]
            x_idx = int(box[0] * self.s / self.img_size)
            y_idx = int(box[1] * self.s / self.img_size)

            # label[y_idx, x_idx, 0] is the response.
            if label[y_idx, x_idx, 0] == 1:
                continue
            label[y_idx, x_idx, 0] = 1
            label[y_idx, x_idx, 5] = 1
            # label[y_idx, x_idx, 10] = 1
            label[y_idx, x_idx, 1: 5] = box
            label[y_idx, x_idx, 6: 10] = box
            # label[y_idx, x_idx, 11: 15] = box
            label[y_idx, x_idx, 10 + cls_idx] = 1
            print(label[y_idx, x_idx])

        new_label = self.label_concatenate(label)
        return new_label, len(objs)

    def label_concatenate(self, label):
        '''
        To change 3-d label to 2-d label.
        :param label: (7, 7, 25)
        :return: new_label: (1, 1470)
        '''
        # class
        print()
        label_class = label[:, :, self.b * 5:]
        label_class_t = label_class.reshape((1, -1))


        # confidence
        label_conf_1 = label[:, :, 0].reshape((7, 7, 1))
        label_conf_2 = label[:, :, 5].reshape((7, 7, 1))
        # label_conf_3 = label[:, :, 10]
        label_conf = np.concatenate([label_conf_1,
                                     label_conf_2,
                                     # label_conf_3,
                                     ],
                                    axis=2)
        label_conf_t = label_conf.reshape((1, -1))


        # box
        label_box_1 = label[:, :, 1: 5]
        label_box_2 = label[:, :, 6: 10]
        # label_box_3 = label[:, :, 11:15]
        label_box = np.concatenate([label_box_1,
                                    label_box_2,
                                    # label_box_3,
                                    ],
                                   axis=2)
        label_box_t = label_box.reshape((1, -1))


        for i in range(7):
            for j in range(7):
                if label_conf[i, j, 0] == 1:
                    print()
                    print(label_conf[i, j, 0])
                    print(label_class[i, j])
                    print(label_box[i, j, :4])


        new_label = np.concatenate([label_class_t, label_conf_t, label_box_t],
                                   axis=1)
        return new_label

    def label_depart(self, label):
        '''
        To change 2-d label back to 3-d label
        :param label: (1, 1715)
        :return: origin_label: (7, 7, 35)
        '''
        boundary1 = self.s * self.s * 20
        boundary2 = boundary1 + self.s * self.s * self.b
        # class
        label_class = np.reshape(label[:, :boundary1], (self.s, self.s, 20))

        # confidence
        label_conf = np.reshape(label[:, boundary1: boundary2], (self.s, self.s, self.b))
        label_conf_1 = label_conf[:, :, 0].reshape((7, 7, 1))
        label_conf_2 = label_conf[:, :, 1].reshape((7, 7, 1))
        # label_conf_3 = label_conf[:, :, 2].reshape((7, 7, 1))

        # box
        label_box = np.reshape(label[:, boundary2:], (self.s, self.s, 4 * self.b))
        label_box_1 = label_box[:, :, : 4]
        label_box_2 = label_box[:, :, 4: 8]
        # label_box_3 = label_box[:, :, 8:]

        origin_label = np.concatenate([label_conf_1,
                                       label_box_1,
                                       label_conf_2,
                                       label_box_2,
                                       # label_conf_3,
                                       # label_box_3,
                                       label_class], axis=2)
        return origin_label
