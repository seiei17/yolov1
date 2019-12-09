import pickle
import numpy as np
import cv2
import os

from data_checking.data_dumping import pascal_voc

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']

def show_pic_and_box(type='or'):
    image, label, imgname = pascal.get()
    img = image[0]

    image = cv2.imread(imgname)

    classes = label[:, :bound1].reshape(7, 7, 20)
    conf = label[:, bound1: bound2].reshape(7, 7, 2)
    response = conf[:, :, 0].reshape(7, 7)
    box = label[:, bound2:].reshape(7, 7, 2, 4)
    coord = box[:, :, 1, :].reshape(7, 7, 4)
    print()

    if type == 'cp':
        img = img[:, ::-1, :]

    # h_ratio = 1.0 * 448 / img.shape[0]
    # w_ratio = 1.0 * 448 / img.shape[1]
    img_h, img_w, _ = image.shape


    # x = (coord[:, :, 0] * img_w / 448).reshape(7, 7)
    # y = (coord[:, :, 1] * img_h / 448).reshape(7, 7)
    # w = ((coord[:, :, 2] * img_w / 448) / 2).reshape(7, 7)
    # h = ((coord[:, :, 3] * img_h / 448) / 2).reshape(7, 7)

    # x = (coord[:, :, 0] * 448 / img_w).reshape(7, 7)
    # y = (coord[:, :, 1] * 448 / img_h).reshape(7, 7)
    # w = ((coord[:, :, 2] * 448 / img_w) / 2).reshape(7, 7)
    # h = ((coord[:, :, 3] * 448 / img_h) / 2).reshape(7, 7)

    x = (coord[:, :, 0]).reshape(7, 7)
    y = (coord[:, :, 1]).reshape(7, 7)
    w = ((coord[:, :, 2]) / 2).reshape(7, 7)
    h = ((coord[:, :, 3]) / 2).reshape(7, 7)

    # img = image

    for i in range(7):
        for j in range(7):
            if response[i, j] == 1:
                print('in the {}row {} col grid.'.format(i, j))
                print('response: ', classes[i, j])
                clxidx = np.argmax(classes[i, j])
                print('class index: ', clxidx)
                clxname = CLASSES[clxidx]
                print('class name: ', clxname )
                print('x, y, w, h: ', x[i, j], y[i, j], w[i, j], h[i, j])
                lu_x = int(x[i, j] - w[i, j])
                lu_y = int(y[i, j] - h[i, j])
                rd_x = int(x[i, j] + w[i, j])
                rd_y = int(y[i, j] + h[i, j])
                print('lu_x, lu_y: ', lu_x, lu_y)
                print('rd_x, rd_y: ', rd_x, rd_y)
                img = cv2.rectangle(img, (lu_x, lu_y), (rd_x, rd_y), (0, 255, 0), 2)

    cv2.imshow('img', img)
    # cv2.imshow('origin', image)

    cv2.waitKey(0)


i = 5
bound1 = 7 * 7 * 20
bound2 = bound1 + 7 * 7 * 2
pascal = pascal_voc('../../../database/', 'train', BatchSize=1)

gt_labels = pascal.gt_labels
gt_labels_cp = pascal.gt_labels_cp

for count in range(len(gt_labels)):
    show_pic_and_box()
    # show_pic_and_box('cp')