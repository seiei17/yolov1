
from yolo_net.yolo_v1_net import yolov1
from utils.pascal_voc_fc import pascal_voc

PASCAL_PATH = '../../database/'

lr = 0.01
# epochs = 15000
BatchSize = 32
num_epoch = 2000
w_decay = 0

net = yolov1(B=2, BatchSize=BatchSize, w_decay=w_decay)
pascal = pascal_voc(PASCAL_PATH=PASCAL_PATH, TrainOrTest='train', BatchSize=BatchSize)
print('------------\nnow begin training\n-------------')
net.train(pascal, epochs=num_epoch, lr=lr)