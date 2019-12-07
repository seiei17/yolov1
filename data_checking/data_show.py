import pickle
import numpy as np
import cv2

from data_checking.data_dumping import pascal_voc

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']

i = 5
bound1 = 7 * 7 * 20
bound2 = bound1 + 7 * 7 * 2
pascal = pascal_voc('../../../database/', 'train')



with open('../../../database/cache/pascal_train_gt_labels.pkl', 'rb') as f:
    gt_labels = pickle.load(f)
    print(gt_labels)
    imgname = gt_labels[0]['imgname']
    label = gt_labels[0]['label']
    classes = label[:, :bound1].reshape(7, 7, 20)
    conf = label[:, bound1: bound2].reshape(7, 7, 2)
    response = conf[:, :, 0].reshape(7, 7)
    box = label[:, bound2: ].reshape(7, 7, 2, 4)
    coord = box[:, :, 1, :].reshape(7, 7, 4)

    img = cv2.imread(imgname)

    h_ratio = 1.0 * 448 / img.shape[0]
    w_ratio = 1.0 * 448 / img.shape[1]

    x = (coord[:, :, 0] / w_ratio).reshape(7, 7)
    y = (coord[:, :, 1] / h_ratio).reshape(7, 7)
    w = (coord[:, :, 2] / w_ratio / 2).reshape(7, 7)
    h = (coord[:, :, 3] / h_ratio / 2).reshape(7, 7)

    cv2.imshow('img', img)

    for i in range(7):
        for j in range(7):
            if response[i, j] == 1:
                print()
                print('response: ', classes[i, j])
                clxidx = np.argmax(classes[i, j])
                print('class index: ', clxidx)
                clxname = CLASSES[clxidx]
                print('class name: ', clxname )
                lu_x = int(x[i, j] - w[i, j])
                lu_y = int(y[i, j] - h[i, j])
                rd_x = int(x[i, j] + w[i, j])
                rd_y = int(y[i, j] + h[i, j])
                print('x_c, y_c: ', lu_x, lu_y)
                print('w, h: ',rd_x, rd_y)

                cv2.rectangle(img, (lu_x, lu_y), (rd_x, rd_y), (0, 255, 0), 5)
                cv2.imshow('img', img)

    cv2.waitKey(0)


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
