from model import YOLOv5
import argparse
import torch
import numpy as np
from pathlib import Path
from utils.loss import ComputeLoss
import torch.backends.cudnn as cudnn
import random
import torch.optim as optim
from torch.cuda import amp
import torch.optim.lr_scheduler as lr_scheduler
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download, is_url
from utils.general import (LOGGER, check_amp, check_dataset, check_file, check_git_status, check_img_size,
                           check_requirements, check_suffix, check_yaml, colorstr, get_latest_run, increment_path,
                           init_seeds, intersect_dicts, labels_to_class_weights, labels_to_image_weights, methods,
                           one_cycle, print_args, print_mutation, strip_optimizer, yaml_save)
#from utils.loggers import Loggers
#from utils.loggers.comet.comet_utils import check_comet_resume
#from utils.loggers.wandb.wandb_utils import check_wandb_resume
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve
from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer,
                               smart_resume, torch_distributed_zero_first)
from copy import deepcopy
import os
import tqdm
import time

WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

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

def train(device, tb_writer=None):
    #logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
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

    opt=argparse.Namespace(weights='.',
                           cfg='D:\\yolo\\yolov5\\models\\yolov5l.yaml',
                           data='D:\\yolo\\yolov5\\data\\RSTD.yaml',
              hyp={'lr0': 0.01, 'lrf': 0.01, 'momentum': 0.937, 'weight_decay': 0.0005, 'warmup_epochs': 3.0,
                   'warmup_momentum': 0.8, 'warmup_bias_lr': 0.1, 'box': 0.05, 'cls': 0.5, 'cls_pw': 1.0, 'obj': 1.0,
                   'obj_pw': 1.0, 'iou_t': 0.2, 'anchor_t': 4.0, 'fl_gamma': 0.0, 'hsv_h': 0.015, 'hsv_s': 0.7,
                   'hsv_v': 0.4, 'degrees': 0.0, 'translate': 0.1, 'scale': 0.5, 'shear': 0.0, 'perspective': 0.0,
                   'flipud': 0.0, 'fliplr': 0.5, 'mosaic': 1.0, 'mixup': 0.0, 'copy_paste': 0.0},
              epochs=300,
              batch_size=8, imgsz=1280, rect=False, resume=False, nosave=False, noval=False, noautoanchor=False,
              noplots=False, evolve=None, bucket='', cache=None, image_weights=False, device='', multi_scale=False,
              single_cls=False, optimizer='SGD', sync_bn=False, workers=8, project='runs\\train', name='exp',
              exist_ok=False, quad=False, cos_lr=False, label_smoothing=0.0, patience=100, freeze=[0], save_period=-1,
              seed=0, local_rank=-1, entity=None, upload_dataset=False, bbox_interval=-1, artifact_alias='latest',
              save_dir='runs\\train\\exp3')

    save_dir = 'D:\\'
    epochs = opt.epochs
    batch_size = opt.batch_size
    weights = ''
    rank = -1
    train_path = 'D:\\prep_rstd\\train'
    val_path = 'D:\\prep_rstd\\val'

    cuda = device != 'cpu'
    init_seeds(2 + rank)

    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay

    nc = 160  # number of classes
    names = [x for x in range(160)]

    # Model

    model = YOLOv5(ch_in=3, nc=nc).to(device)  # create
    #optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), )  # adjust beta1 to momentum
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True)
    lrf = 0.2
    lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - lrf) + lrf  # linear

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    ema = ModelEMA(model) if rank in {-1, 0} else None

    # Resume
    start_epoch, best_fitness = 0, 0.0

    # Image sizes
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    nl = model.detect.nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)
    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Trainloader
    dataloader, dataset = create_dataloader(train_path,
                                            imgsz,
                                            batch_size // WORLD_SIZE,
                                            gs,
                                            single_cls=False,
                                            hyp=hyp,
                                            augment=True,
                                            cache=None if opt.cache == 'val' else opt.cache,
                                            rect=opt.rect,
                                            rank=-1,
                                            workers=8,
                                            image_weights=opt.image_weights,
                                            quad=opt.quad,
                                            prefix=colorstr('train: '),
                                            shuffle=True)
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()

    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    compute_loss = ComputeLoss(model)  # init loss class
    last_opt_step = -1

    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        mloss = torch.zeros(3, device=device)  # mean losses
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)

        optimizer.zero_grad()

        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            t0 = time.perf_counter()
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])


            with amp.autocast(enabled=cuda):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                if rank != -1:
                    loss *= opt.world_size  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.

            # Backward
            scaler.scale(loss).backward()

            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)  # unscale gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Print
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                print(f'{epoch}/{epochs - 1}',
                      i, ' iteration'
                      , mem, *mloss, targets.shape[0], imgs.shape[-1], str(time.perf_counter() - t0) + ' sec/it')

            # end batch ------------------------------------------------------------------------------------------------
        # end epoch ----------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        # Save model
        if not opt.nosave: #or (final_epoch and not opt.evolve):  # if save
            ckpt = {'epoch': epoch,
                    #'best_fitness': best_fitness,
                    #'training_results': results_file.read_text(),
                    'model': deepcopy(de_parallel(model)).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict(),

                    }
        torch.save(ckpt, save_dir + str(epoch) + 'epoch.pt')
        torch.cuda.empty_cache()
    return results


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train('cuda')
