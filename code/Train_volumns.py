#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''

Author   : MG Studio
Datetime : 2018/11/16
Filename : Train_volumns.py
'''


import torch
from torch.utils.data import DataLoader
import os
from Dataloader import *
from Unet3D import *
from viewloss import ViewLoss
# from Visdom import *
from tqdm import tqdm


# 绘制训练曲线
labels1 = ['train_loss']
view1 = ViewLoss(labels=labels1)
# 绘制训练曲线
labels2 = ['val_loss']
view2 = ViewLoss(labels=labels2)

#--------------------------------------------
# 训练参数设置
#----------------------------------------------
os.environ['CUDA_VISIBLE_DEVICE']='0' # 设置使用gpu0
BATCH_SIZE = 1 # 批次大小
EPOCHS = 20 # 迭代轮数
save_model_name = 'unet3D' # 保存训练模型的名字
LR = 1e-2
# loss权重系数
epoch_loss_min = 100


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
    global epoch_loss_min

    if(use_gpu) == True:
        unet = unet.cuda()

    # 创建一个训练数据迭代器
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # 定义优化器
    optimizer = torch.optim.Adam(unet.parameters(), lr=LR)

    # 训练
    epochs = EPOCHS
    for epoch in range(epochs):

        print('开始第{}/{}轮训练'.format(epoch+1, epochs))
        epoch_loss = 0

        # 训练到第epoch轮
        for ii, (datas, labels) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):


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

            loss = loss_function(preds, targets)

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

        # # 一轮结束后，在验证集上进行验证
        # val_loss = val(unet, val_dataset, use_gpu)
        #
        # # 训练一轮后在visdom可视化一些参数
        # print("本轮训练结束：Train_loss: {} Val_loss: {}".format(epoch_loss, val_loss))
        #
        # # 保存模型
        # # save_model(unet, save_model_name)
        #
        # total_loss = epoch_loss * 0.4 + val_loss * 0.6
        # # 保存最佳模型权重
        # if total_loss < epoch_loss_min:
        #     epoch_loss_min = total_loss
        print("本轮训练结束：Train_loss: {}".format(epoch_loss))
        save_model(unet, save_model_name)

        # 保存训练loss数据
        view1.save_loss(csv='./volumes_train.csv')
        # view2.save_loss(csv='./volumes_val.csv')

# -----------------------------------------------------------------
#  模型验证
# -----------------------------------------------------------------
def val(unet, val_dataset, use_gpu=False):
    # 创建一个验证数据集迭代器
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)


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
        loss = loss_function(preds, targets)

        loss_data = loss.data
        print('loss={}'.format(loss_data)) # 打印loss
        view2.losses_hist[0].append(loss_data)
        val_loss += loss_data # 统计loss

    # 把模型恢复为训练模式
    unet.train()

    return val_loss / len(val_dataloader)




if __name__ == '__main__':

    # 實例化一個驗證集對象
    # 訓練集加載
    train_dataset_3d = Datasets_3D(TRAIN_DATA_ROOT)
    print(len(train_dataset_3d))

    # 驗證集加載
    val_dataset_3d = Datasets_3D(VAL_DATA_ROOT)
    print(len(val_dataset_3d))

    # 加載模型
    unet = Unet_3D(in_dim=1, out_dim=4, num_filter=8)
    load_model(unet, save_model_name)

    # 模型訓練
    train(unet, train_dataset_3d, val_dataset_3d, use_gpu=True)
