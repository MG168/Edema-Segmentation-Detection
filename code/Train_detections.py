#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''

Author   : MG Studio
Datetime : 2018/11/16
Filename : Train_detections.py
'''

import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
import csv
import cv2
from viewloss import ViewLoss
from get_data import *
from MyNet import *

seed = 123
np.random.seed(seed)
torch.manual_seed(seed)

TRAIN_CSV_DIR = './train.csv'
VAL_CSV_DIR = './val.csv'
save_model_name = 'detections_03'

Batchsize = 8
Epochs = 20
LR = 5e-4

MEAN = 0.14368844 # 样本均值
STD = 0.08475501 # 样本方差
# MEAN = 0.0 # 样本均值
# STD = 1.0 # 样本方差

# 绘制训练曲线
labels1 = ['train_loss']
view1 = ViewLoss(labels=labels1)
# 绘制训练曲线
labels2 = ['val_loss']
view2 = ViewLoss(labels=labels2)

# 从csv读取文件和标签
def read_csv_file(csv_file_path):

    file_names = []
    file_labels1 = []
    file_labels2 = []
    file_labels3 = []
    with open(csv_file_path, 'r') as files_path:
        path_list = csv.DictReader(files_path)
        fieldnames = path_list.fieldnames
        for path in path_list:
            file_names.append(path[fieldnames[0]])
            file_labels1.append(int(path[fieldnames[1]]))
            file_labels2.append(int(path[fieldnames[2]]))
            file_labels3.append(int(path[fieldnames[3]]))
    return file_names, file_labels1, file_labels2, file_labels3


# 制作Dataloader，將數據加載器封裝起來
class Datasets_detections(Dataset):
    def __init__(self, filepath, csv_file_path):
        self.filepath = filepath
        self.images, self.labels1, self.labels2, self.labels3 = read_csv_file(csv_file_path)
        # 原图预处理
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[MEAN], std=[STD]) # 归一化
        ])

    def __getitem__(self, index):
        image_path = self.images[index]

        img = cv2.imread(image_path, 0)
        # 图片处理：
        # img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA) # 调整大小
        for i in range(1):
            img = cv2.pyrDown(img)  # 图像金字塔下采样

        # img = cv2.equalizeHist(img)  # 直方图均衡化(看情况)
        img = np.expand_dims(img, axis=2)
        img =  self.trans(img)

        label1 = self.labels1[index]
        label2 = self.labels2[index]
        label3 = self.labels3[index]

        label = []
        label.append(label1)
        label.append(label2)
        label.append(label3)

        # 处理成one_hot形式
        yt = torch.LongTensor(label)
        y = torch.unsqueeze(yt, 1)
        yt_onehot = torch.FloatTensor(3, 2)
        yt_onehot.zero_()
        yt_onehot.scatter_(1, y, 1)
        yt_onehot = yt_onehot.view(-1)

        return img, yt_onehot


    def __len__(self):
        return len(self.images)



#--------------------------------------------------------------------
# 模型保存
#--------------------------------------------------------------------
def save_model(model, name = None):
    # 创建checkpoints目录
    if (not (os.path.exists('./checkpoints'))):
        os.mkdir('./checkpoints')
    prefix = './checkpoints/' + name + '.pth'
    # path = prefix + time.strftime('%m%d_%H:%M:%S.pth')
    torch.save(model.state_dict(), prefix)
    # torch.save(model, prefix)
    print('save model successs!!!')



#--------------------------------------------------------------------
# 模型加载
#--------------------------------------------------------------------
def load_model(model, name = None):
    prefix = './checkpoints/' + name + '.pth'
    # path = prefix + time.strftime('%m%d_%H:%M:%S.pth')
    try:
        model.load_state_dict(torch.load(prefix))
        # model = torch.load(prefix)
        print('load model successs!!!')
    except:
        print('load model fault!!!')
        pass


# 批次显示
def show_batch(imgs, BN=False, asgray=False, title='Batch from dataloader'):
    grid = utils.make_grid(imgs)
    img = grid.numpy().transpose((1, 2, 0)) # 通道转换
    if BN:
        img = img * STD + MEAN # 归一化还原
    if asgray:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()


# --------------------------------------------------------------------
#  模型训练
#---------------------------------------------------------------------
def train(model, train_dataloader, val_dataloader, use_gpu = False):

    if (use_gpu) == True:
        model = model.cuda()

    weight = torch.Tensor([1, 1, 1, 2, 1, 20]).cuda()
    loss_func = nn.MultiLabelSoftMarginLoss(weight=weight)

    # loss_func = nn.SoftMarginLoss()

    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-8)

    val_max_acc = 0.0
    for epoch in range(Epochs):
        print('开始第{}/{}轮训练'.format(epoch+1, Epochs))

        losses = []
        acces = []
        # 训练到第epoch轮
        for ii, (datas, labels) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            # 读入训练数据
            inputs, targets = Variable(datas), Variable(labels)

            # 使用GPU计算
            if (use_gpu):
                inputs = inputs.cuda()
                targets = targets.cuda()

            # 前向传播
            output = model(inputs)

            # print(inputs.shape)
            # print(targets.shape)
            # print(output.shape)

            # 梯度清零
            optimizer.zero_grad()

            # 计算loss
            loss = loss_func(output, targets)
            # loss1 = loss_func(output[:, 0:2], targets[:, 0:2])
            # loss2 = loss_func(output[:, 2:4], targets[:, 2:4])
            # loss3 = loss_func(output[:, 4:6], targets[:, 4:6])
            # loss = loss1 * 1 + loss2 * 2 + loss3 * 10  # 设置权重(样本不平衡）

            loss_data = loss.data
            view1.losses_hist[0].append(loss_data)
            losses.append(loss_data)


            # 计算准确率
            _, pred1 = torch.max(output.data[:, 0:2], 1)
            _, pred2 = torch.max(output.data[:, 2:4], 1)
            _, pred3 = torch.max(output.data[:, 4:6], 1)

            _, targ1 = torch.max(targets.data[:, 0:2], 1)
            _, targ2 = torch.max(targets.data[:, 2:4], 1)
            _, targ3 = torch.max(targets.data[:, 4:6], 1)

            # print(output.data[:, 2:4])
            # print(targets.data[:, 2:4])
            # print(pred2)
            # print(targ2)

            train_correct1 = (pred1 == targ1).sum().cpu()
            train_correct2 = (pred2 == targ2).sum().cpu()
            train_correct3 = (pred3 == targ3).sum().cpu()

            acc1 = float(train_correct1.data) / Batchsize
            acc2 = float(train_correct2.data) / Batchsize
            acc3 = float(train_correct3.data) / Batchsize
            acces.append([acc1, acc2, acc3])

            # print('loss={}, acc1={}, acc2={}, acc3={}, '.format(loss_data, acc1, acc2, acc3))

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

        train_loss = np.mean(losses)
        train_acc = np.mean(acces)

        # 一轮结束后，在验证集上进行验证
        val_loss, val_acc = val(model, val_dataloader, use_gpu)

        print('[{}/{}] Train_Loss: {}, ACC: {}'.format(epoch + 1, Epochs, train_loss, train_acc))
        print('[{}/{}] Val_Loss: {}, ACC: {}'.format(epoch + 1, Epochs, val_loss, val_acc))

        # 保存训练loss
        view1.save_loss(csv='./detections_train_3.csv')
        view2.save_loss(csv='./detections_val_3.csv')

        # 保存模型
        # save_model(model, save_model_name)

        # 保存最佳模型权重
        acc = val_acc * 0.6 + train_acc * 0.4 # 分配权重
        if epoch == 0:
            val_max_acc = acc  # 先把第一次记录为acc最大值
            save_model(model, save_model_name) # 保存模型
        else:
            if acc > val_max_acc:
                val_max_acc = acc # 更新最佳的acc
                save_model(model, save_model_name) # 保存模型

# -----------------------------------------------------------------
#  模型验证
# -----------------------------------------------------------------
def val(model, val_dataloader, use_gpu=False):


    # 定义损失函数
    weight = torch.Tensor([1, 1, 1, 2, 1, 20]).cuda()
    loss_func = nn.MultiLabelSoftMarginLoss(weight=weight)
    # loss_func = nn.SoftMarginLoss()

    # 把模型设置为验证模式
    model.eval()

    # 验证
    val_losses = []
    acces = []
    # 训练到第epoch轮
    for ii, (datas, labels) in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
        # 读入训练数据
        inputs, targets = Variable(datas), Variable(labels)

        # 使用GPU计算
        if (use_gpu):
            inputs = inputs.cuda()
            targets = targets.cuda()

        # 前向传播
        output = model(inputs)

        # 计算测试loss
        valloss = loss_func(output, targets)
        # vloss1 = loss_func(output[:, 0:2], targets[:, 0:2])
        # vloss2 = loss_func(output[:, 2:4], targets[:, 2:4])
        # vloss3 = loss_func(output[:, 4:6], targets[:, 4:6])
        # valloss = vloss1 * 1 + vloss2 * 2 + vloss3 * 10  # 设置权重(样本不平衡）


        valloss_data = valloss.data
        view2.losses_hist[0].append(valloss_data)
        val_losses.append(valloss_data)  # 统计loss

        # 计算准确率
        _, pred1 = torch.max(output.data[:, 0:2], 1)
        _, pred2 = torch.max(output.data[:, 2:4], 1)
        _, pred3 = torch.max(output.data[:, 4:6], 1)

        _, targ1 = torch.max(targets.data[:, 0:2], 1)
        _, targ2 = torch.max(targets.data[:, 2:4], 1)
        _, targ3 = torch.max(targets.data[:, 4:6], 1)

        # print(output.data[:, 0:2])
        # print(targets.data[:, 0:2])
        # print(pred1)
        # print(targ1)

        train_correct1 = (pred1 == targ1).sum().cpu()
        train_correct2 = (pred2 == targ2).sum().cpu()
        train_correct3 = (pred3 == targ3).sum().cpu()

        acc1 = float(train_correct1.data) / Batchsize
        acc2 = float(train_correct2.data) / Batchsize
        acc3 = float(train_correct3.data) / Batchsize
        acces.append([acc1, acc2, acc3])

        # print('loss={}, acc1={}, acc2={}, acc3={}, '.format(valloss_data, acc1, acc2, acc3))

    # 把模型恢复为训练模式
    model.train()

    val_loss = np.mean(val_losses)
    val_acc = np.mean(acces)
    return val_loss, val_acc


# 样本直方图分析
def his_analys(csv_file):
    _, l1, l2, l3 = read_csv_file(csv_file)
    l1_numpy = np.asarray(l1, np.float32)
    l2_numpy = np.asarray(l2, np.float32)
    l3_numpy = np.asarray(l3, np.float32)

    plt.subplot(131)
    plt.hist(l1_numpy.ravel(), 2, [0, 2])
    plt.title('REA')
    plt.xticks([0, 1, 2],
               [r'$False$', r'$True$'])
    plt.subplot(132)
    plt.hist(l2_numpy.ravel(), 2, [0, 2])
    plt.title('SRF')
    plt.xticks([0, 1, 2],
               [r'$False$', r'$True$'])
    plt.subplot(133)
    plt.hist(l3_numpy.ravel(), 2, [0, 2])
    plt.title('PED')
    plt.xticks([0, 1, 2],
               [r'$False$', r'$True$'])
    plt.show()









if __name__ == '__main__':

    # his_analys(TRAIN_CSV_DIR) # 直方图分析
    # his_analys(VAL_CSV_DIR)  # 直方图分析

    # 加载数据集
    train_dataset = Datasets_detections(TRAIN_DATA_ROOT, TRAIN_CSV_DIR)
    val_dataset = Datasets_detections(VAL_DATA_ROOT, VAL_CSV_DIR)
    # 打印数据集长度
    print(train_dataset.__len__())
    print(val_dataset.__len__())

    # 制作交叉验证数据集
    n_training_samples = train_dataset.__len__()
    train_sampler = SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))

    n_val_samples = val_dataset.__len__()
    val_sampler = SubsetRandomSampler(np.arange(n_training_samples, n_training_samples + n_val_samples, dtype=np.int64))

    data_set = train_dataset + val_dataset
    print(data_set.__len__())

    train_dataloader = DataLoader(data_set, batch_size=Batchsize, sampler=train_sampler, num_workers=8)
    val_dataloader = DataLoader(data_set, batch_size=Batchsize, sampler=val_sampler, num_workers=8)

    # # 创建一个训练数据迭代器
    # train_dataloader = DataLoader(train_dataset, batch_size=Batchsize, shuffle=True, num_workers=4)
    #
    #
    # # 创建一个验证数据集迭代器
    # val_dataloader = DataLoader(val_dataset, batch_size=Batchsize, shuffle=True, num_workers=4)


    ########################################################################

    # img, label = train_dataset[46]
    # print(img.shape)
    # print(label.shape)


    # # 显示
    # train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True)
    #
    # for i, (batch_x, batch_y) in enumerate(train_dataloader):
    #     if (i < 1):
    #         show_batch(batch_x, BN=True, asgray=True)  # 批量显示数据
    #         show_batch(batch_x, BN=False, asgray=True)  # 批量显示数据
    #         print(batch_y.shape)
    #         print(batch_y)
    #     else:
    #         break


    # load_data(train_dataset)

    ##################################################################################
    # 模型加载
    model = Net3()
    print(model)

    # 加载权重
    load_model(model, save_model_name)

    # 模型训练
    train(model, train_dataloader, val_dataloader, use_gpu=True)
