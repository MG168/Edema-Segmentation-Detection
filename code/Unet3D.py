#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''

Author   : MG Studio
Datetime : 2018/11/16
Filename : Unet3D.py
'''

import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.init as init
import torch.utils.data as data
import torchvision.utils as v_utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np



def conv_block_2_3d(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        conv_block_3d(in_dim,out_dim,act_fn),
        nn.Conv3d(out_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
    )
    return model

def conv_block_3_3d(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        conv_block_3d(in_dim, out_dim, act_fn),
        conv_block_3d(out_dim, out_dim, act_fn),
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
    )
    return model


def maxpool_3d():
    pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
    return pool


def conv_trans_block_3d(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1,output_padding=1),
        nn.BatchNorm3d(out_dim),
        act_fn,
    )
    return model


def conv_block_3d(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        act_fn,
    )
    return model


# 封装 3D Unet 模型
class Unet_3D(nn.Module):

    # 初始化
    def __init__(self, in_dim, out_dim, num_filter):
        super(Unet_3D, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filter = num_filter
        act_fn = nn.LeakyReLU(0.2, inplace=True)

        print("\n------------Initiating U-Net-------------\n")

        self.down_1 = conv_block_2_3d(self.in_dim, self.num_filter, act_fn)
        self.pool_1 = maxpool_3d()
        self.down_2 = conv_block_2_3d(self.num_filter, self.num_filter * 2, act_fn)
        self.pool_2 = maxpool_3d()
        self.down_3 = conv_block_2_3d(self.num_filter * 2, self.num_filter * 4, act_fn)
        self.pool_3 = maxpool_3d()

        self.bridge = conv_block_2_3d(self.num_filter * 4, self.num_filter * 8, act_fn)

        self.trans_1 = conv_trans_block_3d(self.num_filter * 8, self.num_filter * 8, act_fn)
        self.up_1 = conv_block_2_3d(self.num_filter * 12, self.num_filter * 4, act_fn)
        self.trans_2 = conv_trans_block_3d(self.num_filter * 4, self.num_filter * 4, act_fn)
        self.up_2 = conv_block_2_3d(self.num_filter * 6, self.num_filter * 2, act_fn)
        self.trans_3 = conv_trans_block_3d(self.num_filter * 2, self.num_filter * 2, act_fn)
        self.up_3 = conv_block_2_3d(self.num_filter * 3, self.num_filter * 1, act_fn)

        self.bridge = conv_block_3d(self.num_filter, out_dim, act_fn)

    def forward(self, x):
        down_1 = self.down_1(x)
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)

        bridge = self.bridge(pool_3)

        trans_1 = self.trans_1(bridge)
        concat_1 = torch.cat([trans_1, down_3], dim=1)
        up_1 = self.up_1(concat_1)
        trans_2 = self.trans_2(up_1)
        concat_2 = torch.cat([trans_2, down_2], dim=1)
        up_2 = self.up_1(concat_2)
        trans_3 = self.trans_3(up_2)
        concat_3 = torch.cat([trans_3, down_1], dim=1)
        up_3 = self.up_1(concat_3)
        out = self.out(up_3)

        # out = torch.nn.Sigmoid()(out)
        return out


#--------------------------------------------
# Dice_cross计算
#----------------------------------------------
def dice_cross(pred, target):
    smooth = 1. # 平滑项
    iflat = pred.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    dice_coef = (2. * intersection + smooth) / (iflat.sum()+ tflat.sum() + smooth)
    return 1.0 - dice_coef


# 定义损失函数
def loss_function(output, label):
    batch_size, channel, d, h, w = output.size()
    total_loss = 0
    for i in range(batch_size):
        for j in range(d):
            loss = 0
            output_z = output[i:i+1, :, j, :, :]
            label_z = label[i, :, j, :, :]

            # 参考链接：https://blog.csdn.net/hao5335156/article/details/80607732
            # softmax_output_z = nn.Softmax2d()(output_z)
            # logsoftmax_output_z = torch.log(softmax_output_z)
            # loss = nn.NLLLoss()(logsoftmax_output_z, label_z) # 对数似然函数

            logsoftmax_output_z = F.log_softmax(output, dim=1)
            loss = nn.NLLLoss()(logsoftmax_output_z, label_z)  # 對數似然函數

            total_loss += loss

    return total_loss


# 预测一个cube
def pred_cube(output):

    batch_size, channel, d, h, w = output.size()
    result = []
    for i in range(batch_size):
        for j in range(d):
            output_z = output[i:i+1, :, j, :, :]
            print(output_z.shape)

            output_sig = nn.Sigmoid()(output_z).data  # 經過一個sigmoid函數
            print(output_sig.shape)

            P, pred = torch.max(output_sig, dim=1) # 输出分类值和概率

            pred = pred.cpu().numpy()[0]  # 输出体素标记

            result.append(pred)

    return np.asarray(result, np.uint8)




if __name__ == '__main__':

    # 定义一个随机输入，Variable函数相当于存放会变化量的篮子
    x = Variable(torch.ones(1, 1, 16, 256, 128).type_as(torch.FloatTensor())).cuda()
    print('insize:', x.size())

    # model = nn.Conv3d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True)
    # print(model(x).size())

    unet = Unet_3D(in_dim=1, out_dim=4, num_filter=4).cuda()

    output = unet(x)
    print('outsize:', output.size())

    # 模型可视化
    import tensorboardX

    # 定义一个tensorboardX的写对象直接画模型
    # 在工程目录运行cmd，输入：tensorboard --logdir logs
    with tensorboardX.SummaryWriter("./logs/") as writer:
        writer.add_graph(unet, (x))

    # 如果有4个类，可以为每个体素计算标签张量，标签为0,1,2,3个标签

    # label.size() = [batch, class, x, y, z]
    label = Variable(torch.zeros(1, 1, 16, 256, 128).type_as(torch.LongTensor())).cuda()

    label = loss_function(out_put, label)  # 计算loss

    print(loss)

    pred = pred_cube(output)  # 预测结果输出

    print(pred.shape)
