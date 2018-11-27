#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''

搭建unet网络

Author   : MG Studio
Datetime : 2018/11/16
Filename : U_net2.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# [0] 联合卷积模块，（conv->BN->Relu）*2
class union_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(union_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# [1] 初始化卷积模块
class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv,self).__init__()
        self.conv = union_conv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(x)

# [2] down子模块：
class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            union_conv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.mpconv(x)

# [3] up子模块
class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = union_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

# [4] 输出1*1卷积模块
class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.sig(x)
        return x

# [5] 搭建Unet模型
class Unet2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet2, self).__init__()
        self.inc = inconv(in_ch, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 512)

        self.up1 = up(512, 256)
        self.up2 = up(256, 128)
        self.up3 = up(128, 64)
        self.up4 = up(64, 32)
        self.outc = outconv(32, out_ch)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x









if __name__ == '__main__':

    from Dataloader import *
    # 实例化一个训练集对象

    # 加载数据集
    train_dataset = Datasets(TRAIN_DATA_ROOT, s_trans, t_trans)

    Unet = Unet2(1, 3)
    img, msk = train_dataset[0]
    img = Variable(img.unsqueeze(0))
    print(img.shape)
    out = Unet(img)
    print(out.shape)

    from torchviz import make_dot
    # 绘制模型
    g = make_dot(out)
    g.view()

    # 模型可视化
    import tensorboardX

    # 定义一个tensorboardX的写对象直接画模型
    # 在工程目录运行cmd， 输入： tensorboard --logdir logs
    with tensorboardX.SummaryWriter("./logs/") as writer:

        writer.add_graph(Unet, (img))

