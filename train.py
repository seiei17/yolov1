
from yolo_net.yolo_v1_net import yolov1
from utils.pascal_voc_1x1 import pascal_voc

PASCAL_PATH = '../../database/'

lr = 0.001
# epochs = 15000
BatchSize = 32
num_epoch = 0
w_decay = 0.0005
# epochs = [0, 2000, 8000, 12000, 14000, 15000]
#
#
# def lr_reducer(num_epoch):
#     new_lr = lr
#     if num_epoch > epochs[4]:
#         new_lr *= 1e-4
#     elif num_epoch > epochs[3]:
#         new_lr *= 1e-3
#     elif num_epoch > epochs[2]:
#         new_lr *= 1e-2
#     elif num_epoch > epochs[1]:
#         new_lr *= 1e-1
#     return new_lr
#
# net = yolov1(BatchSize=BatchSize)
#
# for i in range(len(epochs) - 1):
#     epoch = epochs[i+1] - epochs[i]
#     init_lr = lr_reducer(num_epoch)
#     pascal = pascal_voc(PASCAL_PATH=PASCAL_PATH, TrainOrTest='train', BatchSize=BatchSize)
#     print('Now begin {}th training: ---------'.format(i+1))
#     net.train(pascal, epoch, init_lr)

net = yolov1(BatchSize=BatchSize, w_decay=w_decay)
pascal = pascal_voc(PASCAL_PATH=PASCAL_PATH, TrainOrTest='train', BatchSize=BatchSize)
print('------------\nnow begin training\n-------------')
net.train(pascal, epochs=200, lr=lr)