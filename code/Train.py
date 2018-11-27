#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''

Author   : MG Studio
Datetime : 2018/11/16
Filename : Train.py
'''



import torch
from torch.utils.data import DataLoader
import os
from Dataloader import *
from U_net import *
from U_net2 import *
from viewloss import ViewLoss
# from Visdom import *
from tqdm import tqdm


# 绘制训练曲线
labels1 = ['train_loss']
view1 = ViewLoss(labels=labels1)
# 绘制训练曲线
labels2 = ['val_loss']
view2 = ViewLoss(labels=labels2)

#----------------------------------------------
# 训练参数设置
#----------------------------------------------
os.environ['CUDA_VISIBLE_DEVICE']='0' # 设置使用gpu0
BATCH_SIZE = 8  # 批次大小
EPOCHS = 5 # 迭代轮数
save_model_name = 'unet1' # 保存训练模型的名字
save_view1_csv_name = './volumes_train1.csv'
save_view2_csv_name = './volumes_val1.csv'
LR = 5e-4
loss_min = 1e2

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
    logsoftmax_output_z = F.log_softmax(output)
    loss = nn.NLLLoss()(logsoftmax_output_z, label)  # 对数似然函数
    return loss

#--------------------------------------------------------------------
# 模型保存
#--------------------------------------------------------------------
def save_model(model, name = None):
    # 创建checkpoints目录
    if (not (os.path.exists('./checkpoints'))):
        os.mkdir('./checkpoints')
    prefix = './checkpoints/' + name + '.pth'
    # path = prefix + time.strftime('%m%d_%H:%M:%S.pth')
    # torch.save(model.state_dict(), path)
    torch.save(model.state_dict(), prefix)
    print('save model successs!!!')


#--------------------------------------------------------------------
# 模型加载
#--------------------------------------------------------------------
def load_model(model, name = None):
    prefix = './checkpoints/' + name + '.pth'
    # path = prefix + time.strftime('%m%d_%H:%M:%S.pth')
    # torch.save(model.state_dict(), path)
    try:
        model.load_state_dict(torch.load(prefix))
        print('load model successs!!!')
    except:
        print('load model fault!!!')
        pass


# --------------------------------------------------------------------
#  模型训练
#---------------------------------------------------------------------
def train(unet, train_dataset, val_dataset, use_gpu = False):
    global loss_min

    if(use_gpu) == True:
        unet = unet.cuda()

    # 创建一个训练数据迭代器
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

    # 定义损失函数
    weight = torch.Tensor([1, 2, 5, 10]).cuda()
    criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=True, weight=weight)  # 交叉熵

    # 定义优化器
    optimizer = torch.optim.Adam(unet.parameters(), lr=LR, weight_decay=1e-8)

    # 训练
    epochs = EPOCHS
    for epoch in range(epochs):

        print('开始第{}/{}轮训练'.format(epoch+1, epochs))
        epoch_loss = 0

        # 训练到第epoch轮
        for ii, (datas,labels) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            
            # 读入训练数据
            inputs, targets = Variable(datas), Variable(labels)


            # 使用GPU计算
            if(use_gpu):
                inputs = inputs.cuda()
                targets = targets.cuda()

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            preds = unet(inputs)


            # 计算loss
            loss = criterion(preds, targets)

            loss_data = loss.data
            print('loss={}'.format(loss_data)) # 打印loss
            view1.losses_hist[0].append(loss_data)
            epoch_loss += loss_data # 统计loss

            # 梯度反向传播
            loss.backward()

            # 参数更新
            optimizer.step()

        # 一轮结束后计算epoch_loss
        epoch_loss = epoch_loss / len(train_dataloader)

        # 一轮结束后，在验证集上进行验证
        # val_loss = val(unet, val_dataset, use_gpu)

        # 训练一轮后在visdom可视化一些参数
        # print("本轮训练结束：Train_loss: {} Val_loss: {}".format(epoch_loss, val_loss))
        print("本轮训练结束：Train_loss: {} ".format(epoch_loss))

        # save_model(unet, save_model_name)

        # 保存最佳模型权重
        # loss_ = epoch_loss * 0.4 + val_loss * 0.6
        # if loss_ < loss_min:
        #     loss_min = loss_
        #     save_model(unet, save_model_name) # 保存模型

        save_model(unet, save_model_name)  # 保存模型

        # 保存训练loss数据
        view1.save_loss(csv=save_view1_csv_name)
        # view2.save_loss(csv=save_view2_csv_name)


# -----------------------------------------------------------------
#  模型验证
# -----------------------------------------------------------------
def val(unet, val_dataset, use_gpu=False):
    # 创建一个验证数据集迭代器
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 定义损失函数
    weight=torch.Tensor([1, 2, 5, 10]).cuda()
    criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=True, weight=weight)  # 交叉熵

    # 把模型设置为验证模式
    unet.eval()

    # 验证
    val_loss = 0
    for ii, (datas, labels) in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
        inputs, targets = Variable(datas), Variable(labels)


        if (use_gpu):
            inputs = inputs.cuda()
            targets = targets.cuda()
        preds = unet(inputs)

        # 计算测试loss
        loss = criterion(preds, targets)

        loss_data = loss.data
        print('loss={}'.format(loss_data)) # 打印loss
        view2.losses_hist[0].append(loss_data)
        val_loss += loss_data # 统计loss

    # 把模型恢复为训练模式
    unet.train()

    return val_loss / len(val_dataloader)


if __name__ == '__main__':


    # # 加载训练集
    train_dataset = Datasets(TRAIN_DATA_ROOT, s_trans, t_trans) \
                    + Datasets(VAL_DATA_ROOT, s_trans, t_trans)

    # 加载测试集
    val_dataset = Datasets(VAL_DATA_ROOT, s_trans, t_trans)


    # # 加载训练集
    # val_dataset = Datasets(TRAIN_DATA_ROOT, s_trans, t_trans)
    #
    # # 加载测试集
    # train_dataset = Datasets(VAL_DATA_ROOT, s_trans, t_trans)

    print(len(train_dataset))
    print(len(val_dataset))

    # 实例化一个 Unet 模型
    unet = Unet(1, 4).cuda()

    # 模型加载
    load_model(unet, save_model_name)

    # 开始训练,使用gpu进行加速
    train(unet, train_dataset, val_dataset, use_gpu=True)

    # 保存模型
    # save_model(unet, save_model_name)
