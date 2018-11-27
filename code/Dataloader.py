#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''

Author   : MG Studio
Datetime : 2018/11/16
Filename : Dataloader.py
'''

import torch
from torchvision import transforms, utils
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt

import cv2
from get_data import *




PATCH_HEIGHT = 512 # 设置图片高
PATCH_WIDTH = 256 # 设置图片宽
MEAN = 0.14368844 # 样本均值
STD = 0.08475501 # 样本方差


# 封装数据集预处理过程，继承 Dataset 属性
class Datasets(Dataset):
    def __init__(self, filepath, source_transform, target_transfrom):
        self.filepath = filepath
        self.images, self.labels = get_img_label_list(self.filepath)
        self.s_transform = source_transform
        self.t_transform = target_transfrom

    def __getitem__(self, index):

        image_path = self.images[index]
        label_path = self.labels[index]

        # 加载原图
        img = cv2.imread(image_path, 0)
        # img = cv2.resize(img, (PATCH_WIDTH, PATCH_HEIGHT), cv2.INTER_AREA)  # 调整大小
        img = cv2.pyrDown(img) # 图像金字塔下采样
        # img = cv2.GaussianBlur(img, (7, 7), 0) # 高斯滤波
        # img = cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21) # 图像去噪
        # img = cv2.equalizeHist(img)# 直方图均衡化
        img = np.expand_dims(img, axis=2)


        # 加载标签
        label = cv2.imread(label_path, 0)
        # label = cv2.resize(label, (PATCH_WIDTH, PATCH_HEIGHT), cv2.INTER_AREA)  # 调整大小
        label = cv2.pyrDown(label) # 图像金字塔下采样
        # mask = np.zeros((label.shape[0], label.shape[1], 3), np.uint8)
        mask = np.zeros((label.shape[0], label.shape[1]), np.int64)
        mask[label == 255] = 1  # REA
        mask[label == 191] = 2  # SRF
        mask[label == 128] = 3  # PED
        mask[label == 0] = 0 # 其他
        # mask = np.expand_dims(mask, axis=2)

        # 预处理操作
        img = self.s_transform(img)
        # mask = self.t_transform(mask)
        mask = torch.from_numpy(mask)
        return img, mask


    def __len__(self):
        return len(self.images)


# 原图预处理
s_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[MEAN], std=[STD]) # 归一化
])

# 标签预处理
t_trans = transforms.Compose([
    # transforms.ToTensor()
])


def convert_label(label, ch=1):
    if ch == 1:
        w, h = label.shape
        mask = np.zeros((w, h), np.uint8)
        mask[label == 0] = 0
        mask[label == 1] = 255
        mask[label == 2] = 191
        mask[label == 3] = 128
        return mask

    else:
        w, h, c = label.shape
        mask = np.zeros((w, h, c), np.uint8)
        mask[label == 0] = 0
        mask[label == 1] = 255
        mask[label == 2] = 191
        mask[label == 3] = 128
        return mask

# 批次显示
def show_batch(imgs, masks, BN=False, title='Batch from dataloader'):
    if BN:
        imgs = imgs * STD + MEAN # 归一化还原
    # print(imgs.shape)
    grid = utils.make_grid(imgs)
    # print(grid.shape)
    img = grid.numpy().transpose((1, 2, 0)) # 通道转换


    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

    masks = masks.unsqueeze(1) # 扩增一个维度
    # print(masks.shape)
    grid2 = utils.make_grid(masks)
    # print(grid2.shape)
    mask = grid2.numpy().transpose((1, 2, 0))  # 通道转换


    print(mask.shape)
    mask = convert_label(mask, ch=3)

    plt.imshow(mask, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()










if __name__ == '__main__':

    # 数据集加载

    train_dataset = Datasets(TRAIN_DATA_ROOT, s_trans, t_trans)
    # train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=4)

    # 实例化一个验证集对象
    val_dataset = Datasets(VAL_DATA_ROOT, s_trans, t_trans)
    # val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)



    # 训练样例分解为训练，测试和交叉验证集
    # Training
    n_training_samples = train_dataset.__len__()
    train_sampler = SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))

    # Validation
    n_val_samples = val_dataset.__len__()
    val_sampler = SubsetRandomSampler(np.arange(n_training_samples, n_training_samples + n_val_samples, dtype=np.int64))
    train_dataloader = DataLoader(train_dataset, batch_size=4, sampler=train_sampler, num_workers=2)
    val_dataloader = DataLoader(train_dataset+val_dataset, batch_size=4, sampler=val_sampler, num_workers=2)





    # 批量显示图片
    for i, (batch_x, batch_y) in enumerate(train_dataloader):
            if (i < 2):
                # print(i, batch_x.size(), batch_y.size())
                show_batch(batch_x, batch_y, BN=False)  # 批量显示数据

            else:
                break

    for i, (batch_x, batch_y) in enumerate(val_dataloader):
        if (i < 2):
            # print(i, batch_x.size(), batch_y.size())
            show_batch(batch_x, batch_y, BN=False)  # 批量显示数据

        else:
            break
