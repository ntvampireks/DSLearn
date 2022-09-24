from components import *
from torch import nn
import torch
import math
from utils.general import xyxy2xywh
from copy import copy
import pandas as pd


class YOLOv5(nn.Module):
    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.detect  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1).detach()  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def __init__(self,  ch_in=3, nc=80):

        self.anchors = [
            [10,13, 16,30, 33,23],  # P3/8
            [30,61, 62,45, 59,119],  # P4/16
            [116,90, 156,198, 373,326]
        ]

        self.nc = nc
        super(YOLOv5, self).__init__()
        #  Backbone
        self.focus = Focus(ch_in=ch_in, ch_out=64, kernel_size=3)

        self.cv1 = Conv(ch_in=64, ch_out=128, kernel_size=3, stride=2)
        self.C3_1 = C3(ch_in=128, ch_out=128, bottleneck_num=3)

        self.cv2 = Conv(ch_in=128, ch_out=256, kernel_size=3, stride=2)
        self.C3_2 = C3(ch_in=256, ch_out=256, bottleneck_num=6)

        self.cv3 = Conv(ch_in=256, ch_out=512, kernel_size=3, stride=2)
        self.C3_3 = C3(ch_in=512, ch_out=512, bottleneck_num=9)

        self.cv4 = Conv(ch_in=512, ch_out=1024, kernel_size=3, stride=2)
        self.C3_4 = C3(ch_in=1024, ch_out=1024, bottleneck_num=3)
        self.spp = SPP(ch_in=1024, ch_out=1024, kernels=[5, 9, 13])

        # neck

        self.nconv1 = Conv(ch_in=1024, ch_out=512, kernel_size=1, stride=1)
        self.upsample1 = nn.Upsample(None, 2, 'nearest')
        self.conc1 = Concat(dimension=1)
        self.nC3_1 = C3(ch_in=1024, ch_out=512, bottleneck_num=3, shortcut=False)

        self.nconv2 = Conv(ch_in=512, ch_out=256, kernel_size=1, stride=1)
        self.upsample2 = nn.Upsample(None, 2, 'nearest')
        self.conc2 = Concat(dimension=1)
        self.nC3_2 = C3(ch_in=512, ch_out=256, bottleneck_num=3, shortcut=False)

        self.nconv3 = Conv(ch_in=256, ch_out=256, kernel_size=3, stride=2)
        self.conc3 = Concat(dimension=1)
        self.nC3_3 = C3(ch_in=512, ch_out=512, bottleneck_num=3, shortcut=False)

        self.nconv4 = Conv(ch_in=512, ch_out=512, kernel_size=3, stride=2)
        self.conc4 = Concat(dimension=1)
        self.nC3_4 = C3(ch_in=1024, ch_out=1024, bottleneck_num=3, shortcut=False)

        self.detect = Detect(nc=nc, anchors=self.anchors, ch=[256, 512, 1024], inplace=True)
        # детектор
        s = 256  # 2x min stride
        self.detect.inplace = True
        self.detect.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.empty(1, ch_in, s, s))])  # forward
        #check_anchor_order(m)  # must be in pixel-space (not grid-space)
        self.detect.anchors /= self.detect.stride.view(-1, 1, 1)
        self.stride = self.detect.stride
        self._initialize_biases()  # only run once

    def forward(self, x):
        lst_backbone = []
        x = self.focus(x)
        x = self.cv1(x)
        x = self.C3_1(x)

        x = self.cv2(x)
        x = self.C3_2(x)

        lst_backbone.append(x)  # P3

        x = self.cv3(x)
        x = self.C3_3(x)

        lst_backbone.append(x)  # P4

        x = self.cv4(x)
        x = self.C3_4(x)
        x = self.spp(x)

        #lst_backbone.append(x)  # P5

        lst_neck = []
        x = self.nconv1(x)

        lst_neck.append(x)

        x = self.upsample1(x)
        x = self.conc1([lst_backbone[1], x])
        x = self.nC3_1(x)
        x = self.nconv2(x)

        lst_neck.append(x)

        x = self.upsample2(x)
        x = self.conc2([lst_backbone[0], x])
        x = self.nC3_2(x)

        #lst_neck.append(x)

        to_detect = [None, None, None]

        to_detect[0] = x

        x = self.nconv3(x)
        x = self.conc3([lst_neck[1], x])
        x = self.nC3_3(x)

        to_detect[1] = x

        x = self.nconv4(x)
        x = self.conc4([lst_neck[0], x])

        to_detect[2] = self.nC3_4(x)

        return self.detect(to_detect)


class Detections:
    # YOLOv5 detections class for inference results
    def __init__(self, ims, pred, files, times=(0, 0, 0), names=None, shape=None):
        super().__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d) for im in ims]  # normalizations
        self.ims = ims  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.times = times  # profiling times
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)

        self.s = shape  # inference BCHW shape

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        r = range(self.n)  # iterable
        x = [Detections([self.ims[i]], [self.pred[i]], [self.files[i]], self.times, self.names, self.s) for i in r]
        # for d in x:
        #    for k in ['ims', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
        #        setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n  # override len(results)



