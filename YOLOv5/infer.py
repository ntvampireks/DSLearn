from model import YOLOv5, Detections
import torch
from utils.general import non_max_suppression
import numpy as np
import cv2
from utils.augmentations import letterbox
import pandas as pd
names ={
  0: 'person',
  1: 'bicycle',
  2: 'car',
  3: 'motorcycle',
  4: 'airplane',
  5: 'bus',
  6: 'train',
  7: 'truck',
  8: 'boat',
  9: 'traffic light',
  10: 'fire hydrant',
  11: 'stop sign',
  12: 'parking meter',
  13: 'bench',
  14: 'bird',
  15: 'cat',
  16: 'dog',
  17: 'horse',
  18: 'sheep',
  19: 'cow',
  20: 'elephant',
  21: 'bear',
  22: 'zebra',
  23: 'giraffe',
  24: 'backpack',
  25: 'umbrella',
  26: 'handbag',
  27: 'tie',
  28: 'suitcase',
  29: 'frisbee',
  30: 'skis',
  31: 'snowboard',
  32: 'sports ball',
  33: 'kite',
  34: 'baseball bat',
  35: 'baseball glove',
  36: 'skateboard',
  37: 'surfboard',
  38: 'tennis racket',
  39: 'bottle',
  40: 'wine glass',
  41: 'cup',
  42: 'fork',
  43: 'knife',
  44: 'spoon',
  45: 'bowl',
  46: 'banana',
  47: 'apple',
  48: 'sandwich',
  49: 'orange',
  50: 'broccoli',
  51: 'carrot',
  52: 'hot dog',
  53: 'pizza',
  54: 'donut',
  55: 'cake',
  56: 'chair',
  57: 'couch',
  58: 'potted plant',
  59: 'bed',
  60: 'dining table',
  61: 'toilet',
  62: 'tv',
  63: 'laptop',
  64: 'mouse',
  65: 'remote',
  66: 'keyboard',
  67: 'cell phone',
  68: 'microwave',
  69: 'oven',
  70: 'toaster',
  71: 'sink',
  72: 'refrigerator',
  73: 'book',
  74: 'clock',
  75: 'vase',
  76: 'scissors',
  77: 'teddy bear',
  78: 'hair drier',
  79: 'toothbrush',
}

hyp = {'lr0': 0.01,
       'lrf': 0.01,
       'momentum': 0.937,
       'weight_decay': 0.0005,
       'warmup_epochs': 3.0,
       'warmup_momentum': 0.8,
       'warmup_bias_lr': 0.1,
       'box': 0.05,
       'cls': 0.5,
       'cls_pw': 1.0,
       'obj': 1.0,
       'obj_pw': 1.0,
       'iou_t': 0.2,
       'anchor_t': 4.0,
       'fl_gamma': 0.0,
       'hsv_h': 0.015,
       'hsv_s': 0.7,
       'hsv_v': 0.4,
       'degrees': 0.0,
       'translate': 0.1,
       'scale': 0.5,
       'shear': 0.0,
       'perspective': 0.0,
       'flipud': 0.0,
       'fliplr': 0.5,
       'mosaic': 1.0,
       'mixup': 0.0,
       'copy_paste': 0.0}

def check_img(ckpt, img_path, ch_in, nc, img_size, device='cuda'):
    '''

    :param ckpt:        путь к файлу с сохраненной конфигурацией
    :param img_path:    пусть к изорбражению
    :param ch_in:       количество слоев детекции
    :param nc:          количество определяемых классов
    :param img_size:    размер изображения, в квадрат с такой стороной в пикселях будут вписаны изображения
                        тренировочной выборки
    :return:
    '''
    ckpt = torch.load(ckpt, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
    model = YOLOv5(ch_in=ch_in, nc=nc).to(device)
    model.load_state_dict(ckpt['model'].float().state_dict())  # create
    model.eval()
    gs = max(int(model.stride.max()), 32)

    im0 = cv2.imread(img_path)  # BGR
    im = letterbox(im0, img_size, stride=gs, auto=True)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous

    im = torch.from_numpy(im).float().to(device)
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    im /= 255  # 0 - 255 to 0.0 - 1.0
    pred = model(im)[0]
    pred = non_max_suppression(pred, conf_thres=0.3, iou_thres=0.3, )
    d = Detections(im, pred, img_path, names=names)

    return pd.concat(d.pandas().xyxy)

if __name__ == '__main__':
    x = check_img('F:\\325epoch.pt', 'D:\\t1\\000000000025.jpg', 3, 80, 640, device='cuda')
    print(x)