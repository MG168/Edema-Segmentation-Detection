#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''

Author   : MG Studio
Datetime : 2018/11/16
Filename : MyNet.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import *
from torchvision.models.densenet import *

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(1, 32, 3, 1, 1),
#             nn.ReLU(),
#             nn.MaxPool2d(2))
#         self.conv2 = torch.nn.Sequential(
#             nn.Conv2d(32, 64, 3, 1, 1),
#             nn.ReLU(),
#             nn.MaxPool2d(2))
#         self.conv3 = torch.nn.Sequential(
#             nn.Conv2d(64, 64, 3, 1, 1),
#             nn.ReLU(),
#             nn.MaxPool2d(2))
#
#         self.dense = torch.nn.Sequential(
#             nn.Linear(64 * 32 * 16, 2048),
#             nn.Dropout(0.5),
#             nn.ReLU(),
#             nn.Linear(2048, 512),
#             nn.Dropout(0.5),
#             nn.ReLU(),
#             nn.Linear(512, 2),
#             nn.Sigmoid())
#
#
#     def forward(self, x):
#         c1 = self.conv1(x)
#         c2 = self.conv2(c1)
#         c3 = self.conv3(c2)
#         # print(c3.shape)
#         res = c3.view(c3.size(0), -1)
#
#         out = self.dense(res)
#         return out
#
#

# # 检测网络
# def Net():
#     model = resnet50()
#
#     for parma in model.parameters():
#         parma.requires_grad = True # 重新训练
#
#     model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2),
#                                   padding=(3, 3), bias=False) # 改写第一个卷积层
#     model.fc = torch.nn.Sequential(
#                 nn.Linear(40960, 1024),
#                 nn.Dropout(0.5),
#                 nn.ReLU(),
#                 nn.Linear(1024, 2),
#                 nn.Sigmoid() #
#                 ) # 改写全链接层层
#
#     return model



# 检测网络, 基于Resnet50
def Net():
    model = resnet50()

    for parma in model.parameters():
        parma.requires_grad = True # 重新训练

    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2),
                                  padding=(3, 3), bias=False) # 改写第一个卷积层


    model.fc = torch.nn.Sequential(
        nn.Linear(40960, 2048),
        nn.Dropout(0.5),
        nn.ReLU(),
        nn.Linear(2048, 512),
        nn.Dropout(0.5),
        nn.ReLU(),
        nn.Linear(512, 6),
        # nn.Sigmoid()
    )
    return model


# 检测网络2，基于Densenet161
def Net2():
    model = densenet161()

    for parma in model.parameters():
        parma.requires_grad = True # 重新训练

    model.features.conv0 = torch.nn.Conv2d(1, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # 改写第一个卷积层

    # model.classifier = nn.Linear(in_features=44160, out_features=6, bias=True)
    model.classifier = torch.nn.Sequential(
                                            nn.Linear(44160, 2048),
                                            nn.ReLU(),
                                            nn.Dropout(0.5),
                                            nn.Linear(2048, 512),
                                            nn.ReLU(),
                                            nn.Dropout(0.5),
                                            nn.Linear(512, 6),
                                            # nn.Sigmoid()
                                            )

    return model



# 检测网络, 基于Resnet101
def Net3():
    model = resnet101()

    for parma in model.parameters():
        parma.requires_grad = True # 重新训练

    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2),
                                  padding=(3, 3), bias=False) # 改写第一个卷积层

    model.fc = torch.nn.Sequential(
        nn.Linear(40960, 2048),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 6),
        # nn.Sigmoid()
    )
    return model




if __name__ == '__main__':


    # 定义一个随机输入
    x = Variable(torch.rand(1, 1, 512, 256))

    # 实例化模型
    model = Net3()
    print(model)

    # 模型输出
    out = model(x)
    print(out)

    # 模型可视化
    # from visualize import make_dot
    #
    # g = make_dot(out)
    # g.view()

    from tensorboardX import SummaryWriter

    with SummaryWriter(log_dir='./logs') as writer:
        writer.add_graph(model, (x))
