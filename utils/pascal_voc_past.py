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
    def __init__(self, PASCAL_PATH, TrainOrTest, BatchSize, S=7, FLIPPED=True, REBUILD=False):
        self.pascal_path = PASCAL_PATH
        self.devkit_path = os.path.join(self.pascal_path, 'VOCdevkit')
        self.data_path = os.path.join(self.devkit_path, 'VOC2007')
        self.cache_path = os.path.join(self.pascal_path, 'cache')
        self.img_size = 448
        self.cls = CLASSES
        self.BatchSize = BatchSize
        self.s = S
        self.cls_to_idx = dict(zip(self.cls, range(len(self.cls))))
        self.flipped = FLIPPED
        self.rebuild = REBUILD
        self.TrainOrTest = TrainOrTest
        self.cursor = 0
        self.epoch = 1
        self.gt_labels = None
        self.prepare()

    def get_len(self):
        return len(self.gt_labels)

    def generator(self):
        while True:
            images, labels = self.get()
            print(images.shape)
            print(labels.shape)
            yield images, labels

    def get(self):
        images = np.zeros((self.BatchSize, self.img_size, self.img_size, 3))
        labels = np.zeros((self.BatchSize, self.s, self.s, 25))
        count = 0
        while count < self.BatchSize:
            imgname = self.gt_labels[self.cursor]['imgname']
            flipped = self.gt_labels[self.cursor]['flipped']
            images[count, :, :, :] = self.image_read(imgname, flipped)
            labels[count, :, :, :] = self.gt_labels[self.cursor]['label']
            count += 1
            self.cursor += 1
            if self.cursor >= len(self.gt_labels):
                np.random.shuffle(self.gt_labels)
                self.cursor = 0
                self.epoch += 1
        return images, labels

    def image_read(self, imgname, flipped=False):
        image = cv2.imread(imgname)
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image / 255.0) * 2.0 - 1.0
        if flipped:
            image = image[:, ::-1, :]
        return image

    def prepare(self):
        gt_labels = self.load_labels()
        if self.flipped:
            print('Appending horizontally-flipped training examples ...')
            gt_labels_cp = copy.deepcopy(gt_labels)
            for idx in range(len(gt_labels_cp)):
                gt_labels_cp[idx]['flipped'] = True
                gt_labels_cp[idx]['label'] = gt_labels_cp[idx]['label'][:, ::-1, :]
                for i in range(self.s):
                    for j in range(self.s):
                        if gt_labels_cp[idx]['label'][i, j, 0] == 1:
                            # change the x's coord
                            gt_labels_cp[idx]['label'][i, j, 1] = self.s - 1 -\
                                                                  gt_labels_cp[idx]['label'][i, j, 1]
            gt_labels += gt_labels_cp
        np.random.shuffle(gt_labels)
        self.gt_labels = gt_labels
        return gt_labels

    def load_labels(self):
        cache_file = os.path.join(self.cache_path,
                                  'pascal_' +
                                  self.TrainOrTest +
                                  '_gt_labels.pkl')
        # if cache file is existing and no need to rebuild,
        # then just load pkl file.
        if os.path.isfile(cache_file) and not self.rebuild:
            print('Loading gt_labels from: ' + cache_file)
            with open(cache_file, 'rb') as cachefile:
                gt_labels = pickle.load(cachefile)
            return gt_labels

        print('Processing gt_labels from: ' + self.data_path)

        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        if self.TrainOrTest == 'train':
            txtname = os.path.join(self.data_path, 'ImageSets', 'Main', 'trainval.txt')
        else:
            txtname = os.path.join(self.data_path, 'ImageSets', 'Main', 'val.txt')
        with open(txtname, 'r') as file:
            self.image_index = [x.strip() for x in file.readlines()]

        gt_labels = []
        for index in self.image_index:
            label, num = self.load_pascal_annotation(index)
            if num == 0:
                continue
            imgname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
            gt_labels.append({'imgname': imgname,
                              'label': label,
                              'flipped': False})
        print('Saving gt_labels to: ' + cache_file)
        with open(cache_file, 'wb') as file:
            pickle.dump(gt_labels, file)
        return gt_labels

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

        label = np.zeros((self.s, self.s, 25))
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
            # label[y_idx, x_idx, 5] = 1
            label[y_idx, x_idx, 1: 5] = box
            # label[y_idx, x_idx, 6: 10] = box
            label[y_idx, x_idx, 5 + cls_idx] = 1

        return label, len(objs)