from model import YOLOv5
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import random
import torch.optim.lr_scheduler as lr_scheduler
from utils.dataloaders import create_dataloader
from utils.general import check_img_size, \
                            colorstr, \
                            init_seeds
from utils.loss import ComputeLoss
from utils.torch_utils import de_parallel
from copy import deepcopy
import time


def init_torch_seeds(seed=0):
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)
    if seed == 0:  # slower, more reproducible
        cudnn.benchmark, cudnn.deterministic = False, True
    else:  # faster, less reproducible
        cudnn.benchmark, cudnn.deterministic = True, False


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    init_torch_seeds(seed)


def train(device, ch_in, nc, imgsz,  weights, train_path, val_path, batch_size, epochs, save_dir):
    '''
    :param device:      устройство, cuda или cpu
    :param ch_in:       количество каналов в изображении и слоев обнаружения
    :param nc:          количество определяемых классов
    :param imgsz:       размер изображения, в квадрат с такой стороной в пикселях будут вписаны изображения
                        тренировочной выборки
    :param weights:     путь к файлу с сохраненной конфигурацией
    :param train_path:  путь к папке с тренировочной выборкой, структура папок должна соответствовать
                        описанной здесь: https://docs.ultralytics.com/tutorials/train-custom-datasets/
    :param val_path:    путь к папке с валидационной выборкой, структура папок должна соответствовать
                        описанной здесь: https://docs.ultralytics.com/tutorials/train-custom-datasets/
    :param batch_size:  размер пачки. На GeForce RTX 3090 использует порядка 17 ГБ памяти при значении 16.
                        Можно уменьшить если использовать половинную точность или AMP
    :param epochs:      количество эпох обучения
    :param save_dir:    папка для сохранения чекпойнтов
    :return:
    '''
    # logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    hyp = {'lr0': 0.005,
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
    names = {
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

    rank = -1
    lrf = hyp['lrf']
    lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - lrf) + lrf  # linear
    init_seeds(2 + rank)

    # Model
    if weights:
        ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
        model = YOLOv5(ch_in=ch_in, nc=nc).to(device)
        model.load_state_dict(ckpt['model'].float().state_dict())    # create
        optimizer = torch.optim.SGD(model.parameters(), lr=hyp['lr0'], momentum=hyp['momentum'],
                                    weight_decay=hyp['weight_decay'], nesterov=True)
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch']
    else:
        model = YOLOv5(ch_in=ch_in, nc=nc).to(device)
        start_epoch, best_fitness = 0, 0.0
        optimizer = torch.optim.SGD(model.parameters(), lr=hyp['lr0'], momentum=hyp['momentum'],
                                    weight_decay=hyp['weight_decay'], nesterov=True)

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # Image sizes
    gs = max(int(model.stride.max()), 32)               # вычисляем размер сетки
    nl = model.detect.nl                                # количество слоев обнаружения
    imgsz = check_img_size(imgsz, gs, floor=gs * 2)     # приведение к новому размеру, если требуется

    # Trainloader # использовал готовые из библиотеки YOLO
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size=batch_size, stride=gs,
                                            single_cls=False, hyp=hyp,
                                            cache=None, rect=False,
                                            workers=8, image_weights=False,
                                            quad=False, prefix=colorstr('train: '), shuffle=True)

    val_loader = create_dataloader(val_path, imgsz, batch_size=batch_size//2, stride=gs,
                                            single_cls=False, hyp=hyp,
                                            cache=None, rect=False,
                                            workers=8, image_weights=False,
                                            quad=False, prefix=colorstr('val: '), shuffle=True)[0]

    # складываем внутрь модели параметры, чтоб при загрузке можно было посмотреть.
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = False
    model.nc = nc
    model.hyp = hyp
    model.gr = 1.0
    model.names = names

    # Start training
    scheduler.last_epoch = start_epoch - 1
    compute_loss = ComputeLoss(model)
    val_loss = torch.zeros(3, device=device)
    l = torch.zeros(1, device=device)
    t0 = time.perf_counter()
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        val_loss = torch.zeros(3, device=device)
        mloss = torch.zeros(3, device=device)  # mean losses
        l = torch.zeros(1, device=device)
        model.train()
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in enumerate(dataloader):  # batch --------------------------------------------
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  #
            pred = model(imgs)  # forward
            loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
            loss.backward()
            if i % 2 == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
                optimizer.zero_grad()
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            if i % 1 == 0:
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                #lbox, lobg, lcls, loss
                print(f'{epoch}/{epochs - 1}',
                      i, ' iteration',
                      mem, *mloss, loss,  targets.shape[0], imgs.shape[-1], str(time.perf_counter() - t0) + ' sec/it')
                t0 = time.perf_counter()
            # end batch ------------------------------------------------------------------------------------------------
        # end epoch ----------------------------------------------------------------------------------------------------

        scheduler.step()
        # validate
        torch.cuda.empty_cache()
        model.eval()
        for batch_i, (im, targets, paths, shape) in enumerate(val_loader):
            im = im.to(device, non_blocking=True).float()
            targets = targets.to(device)
            nb, _, height, width = im.shape  # batch size, channels, height, width
            with torch.no_grad():
                out, train_out = model(im) if compute_loss else (model(im, augment=False), None)
                loss, loss_items = compute_loss(train_out, targets.to(device))
                val_loss += loss_items
                l += loss

        val_loss = (val_loss/len(val_loader)).tolist()  # выдает дичь, но после обучения детектит нормально
        l = (l / len(val_loader)).tolist()

        # lbox, lobg, lcls, loss
        print(*val_loss, l)
        if epoch % 5 == 0 or epoch == epochs:
            ckpt = {'epoch': epoch,
                    'model': deepcopy(de_parallel(model)),
                    'optimizer': optimizer.state_dict(),
                    }
            torch.save(ckpt, save_dir + str(epoch) + 'epoch.pt')
    torch.cuda.empty_cache()
    return *val_loss, l


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train(device='cuda',
          ch_in=3,
          nc=80,
          imgsz=640,
          weights=None,
          train_path='D:\\coco128\\images\\train2017',
          val_path='D:\\coco128v\\images\\val2017',
          batch_size=16,
          epochs=600,
          save_dir='F:\\')
