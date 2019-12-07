# test

from yolo_net.yolo_v1_net import yolov1
from yolo_net.yolo_v1_utils import YoloDetector


def main():

    yolo = yolov1(S=7, B=3, num_classes=20, BatchSize=1)
    YOLOv1_weight = './history/YOLOv1_weight.h5'
    detector = YoloDetector(yolo, YOLOv1_weight, 20, 448, S=7, B=2)

    # detect from camera
    # cap = cv2.VideoCapture(-1)
    # detector.camera_detector(cap)

    # detect from image file
    imname = 'test/person.jpg'
    detector.image_detector(imname)