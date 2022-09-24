import torch
from torch import nn


def autopad(k, p=None):  # kernel, padding
    # если паддинг не задан - то считаем его как пололвину от размера ядра свертки
    # иначе возвращаем его как есть
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """
    Реализация слоя CBL для сети YOLOv5
    """
    def __init__(self, ch_in, ch_out, kernel_size=1, stride=1, padding=None, groups=1, eps=1e-3,
                        momentum=0.03):
        """ конструктор слоя
        :param ch_in: количество входных каналов
        :param ch_out: количество каналов после свертки
        :param kernel_size: размер ядра свертки
        :param stride: страйд свертки
        :param padding: размер паддинга
        :param groups:
        """
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=ch_in,
                              out_channels=ch_out,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=autopad(kernel_size, padding),
                              groups=groups,
                              bias=False)
        self.bn = nn.BatchNorm2d(ch_out, eps=eps, momentum=momentum)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)


class Focus(nn.Module):
    """
    Реализация слоя Focus для сети YOLOv5
    """
    def __init__(self, ch_in, ch_out, kernel_size=1, stride=1, padding=None, groups=1, eps=1e-3, momentum=0.03):
        """ конструктор слоя
        :param ch_in: количество входных каналов
        :param ch_out: количество каналов после свертки
        :param kernel_size: размер ядра свертки
        :param stride: страйд свертки
        :param padding: размер паддинга
        :param groups:
        """
        super(Focus, self).__init__()
        self.conv = Conv(ch_in=ch_in*4, # четыре потому что исходное изображение режется на 4 части в итоге - 12 каналов
                        ch_out=ch_out,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1,
                        groups=1,
                        eps=1e-3,
                        momentum=0.03)

    def forward(self, x):
        # преобразуем тензор с изображениям batch x 3 x W x H -> batch x 12 x (W/2) x (H/2)
        x1 = x[..., ::2, ::2]  # с нулевого пикселя по x с шагом 2, с нулевого пикселя по y с шагом 2
        x2 = x[..., 1::2, ::2]  # с первого пикселя по x с шагом 2, с нулевого пикселя по y с шагом 2
        x3 = x[..., ::2, 1::2]  # с нулевого пикселя по x с шагом 2, с первого пикселя по y с шагом 2
        x4 = x[..., 1::2, 1::2]  # с первого пикселя по x с шагом 2, с первого пикселя по y с шагом 2
        # конкатенируем. можно было в одну строчку, но так понятней
        x = torch.cat([x1, x2, x3, x4], 1)
        return self.conv(x)


class Bottleneck(nn.Module):
        # Standard bottleneck
    def __init__(self, ch_in, ch_out, shortcut=True, groups=1, expansion=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        """конструктор слоя
        :param ch_in: количество входных каналов
        :param ch_out: количество каналов после свертки
        :param shortcut: добалять ли вход к выходу
        :param groups:
        :param expansion: масштабирование для скрытого слоя свертки
        """
        super(Bottleneck, self).__init__()
        c_ = int(ch_out * expansion)  # число каналов свертки для скрытого слоя
        self.cv1 = Conv(ch_in=ch_in, ch_out=c_, kernel_size=1, stride=1)
        self.cv2 = Conv(ch_in=c_, ch_out=ch_out, kernel_size=3, stride=1, groups=groups)
        self.add = shortcut and ch_in == ch_out

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, ch_in, ch_out, bottleneck_num=1, shortcut=True, groups=1, expansion=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(ch_out * expansion)  # число каналов свертки для скрытого слоя
        self.cv1 = Conv(ch_in=ch_in, ch_out=c_, kernel_size=1, stride=1)
        self.cv2 = Conv(ch_in=ch_in, ch_out=c_, kernel_size=1, stride=1)
        self.cv3 = Conv(ch_in=2 * c_, ch_out=ch_out, kernel_size=1)  # act=FReLU(c2)
        self.bottleneck = nn.Sequential(*[Bottleneck(ch_in=c_, ch_out=c_, shortcut=shortcut,
                                            groups=groups, expansion=1.0) for _ in range(bottleneck_num)])


    def forward(self, x):
        # громоздко, но отражает более наглядно структуру вызовов
        r1 = self.cv1(x)
        r1 = self.bottleneck(r1)
        r2 = self.cv2(x)
        r2 = torch.cat((r1,r2), dim=1)
        return self.cv3(r2)

class SPP(nn.Module):
    def __init__(self, ch_in, ch_out, kernels=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = ch_in // 2  # hidden channels
        self.cv1 = Conv(ch_in, c_, 1, 1)
        # так как применяются n слоев MaxPool - то количество каналов кратно количеству пулингов
        self.cv2 = Conv(c_ * (len(kernels) + 1), ch_out, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in kernels])

    def forward(self, x):
        #громоздко, но на мой взгляд так понятей
        x = self.cv1(x)
        r = [m(x) for m in self.m]
        r = torch.cat([x] + r, 1)
        return self.cv2(r)


class Concat(nn.Module):
    # Конкатенация списка тензоров по указанному измерению
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.dimension = dimension

    def forward(self, x):
        return torch.cat(x, self.dimension)

class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid

        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic or self.grid[i].device != x[i].device:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if self.inplace:
                    r = y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)
                    y[..., 0:2] = r * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.na, 1, 1, 2)  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
