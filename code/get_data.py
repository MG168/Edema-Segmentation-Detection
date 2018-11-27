#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''

Author   : MG Studio
Datetime : 2018/11/16
Filename : get_data.py
'''


import os
import numpy as np
import cv2
from tqdm import tqdm
import csv



DATA_ROOT = "../data/"
TRAIN_DATA_ROOT = DATA_ROOT + "Edema_trainingset/"
VAL_DATA_ROOT = DATA_ROOT + "Edema_validationset/"
TEST_DATA_ROOT = DATA_ROOT + "Edema_testset/"



# 解析安装包内的数据
def get_cube_list(filepath, have_label=True):

    if have_label:
        cube_img_names = os.listdir(filepath + 'original_images/')
        cube_mask_names = os.listdir(filepath + 'label_images/')
        cube_img_names = sorted(cube_img_names)
        cube_mask_names = sorted(cube_mask_names)
        cube_img_label_list = []
        for i, l in zip(cube_img_names, cube_mask_names):
            cube_img_path = filepath + 'original_images/' + i + '/'
            cube_label_path = filepath + 'label_images/' + l + '/'
            cube_img_label_list.append([cube_img_path, cube_label_path])

        return cube_img_label_list

    else:
        cube_img_names = os.listdir(filepath + 'original_images/')
        cube_img_names = sorted(cube_img_names)
        cube_img_list = []
        for i in cube_img_names:
            cube_img_path = filepath + 'original_images/' + i + '/'
            cube_img_list.append(cube_img_path)

        return cube_img_list

# 从zip解析列表
def get_file_list(imgcubepath, labelcubepath=None):
    if labelcubepath != None:
        img_list = os.listdir(imgcubepath)
        label_list = os.listdir(labelcubepath)
        img_list = sorted(img_list, key=lambda x:int(x.split('.')[0])) # 排序
        label_list = sorted(label_list, key=lambda x: int(x.split('.')[0])) # 排序
        file_list = []
        for img, label  in zip(img_list, label_list):
            img_file = imgcubepath + img
            label_file = labelcubepath + label
            file_list.append([img_file, label_file])

        return file_list

    else:
        img_list = os.listdir(imgcubepath)
        img_list = sorted(img_list, key=lambda x: int(x.split('.')[0]))
        file_list = []
        for img in img_list:
            img_file = imgcubepath + img
            file_list.append(img_file)

        return file_list

# 获取文件列表
def get_file_lists(filepath):
    id_paths = os.listdir(filepath + 'original_images/')
    original_images_list = []
    label_images_list = []
    for id_path in id_paths:
        id_name =  id_path.split('.')[0]
        original_images_list.append('original_images/' + id_name + '.img')
        label_images_list.append('label_images/' + id_name + '_labelMark')

    return original_images_list, label_images_list

# 加载 img 和 label 路径
def get_img_label_list(filepath, have_label=True):

    if have_label:
        images_list, labels_list = [], []
        cube_img_label_list = get_cube_list(filepath, have_label=True)
        for cube_list in cube_img_label_list:
            file_list = get_file_list(cube_list[0], cube_list[1])
            for file_path in file_list:
                images_list.append(file_path[0])
                labels_list.append(file_path[1])
        return images_list, labels_list


    else:
        images_list = []
        cube_img_list = get_cube_list(filepath, have_label=have_label)
        for cube_list in cube_img_list:
            file_list = get_file_list(cube_list)
            for file_path in file_list:
                images_list.append(file_path)

        return images_list


# 数据转 csv 文件：
def data_to_csv(filepath, mode='detection', csv_dir='train.csv'):
    images_list, labels_list = get_img_label_list(filepath, have_label=True)
    # print(images_list)
    # print(labels_list)
    if mode == 'detection':
        list = []
        list.append(['File Name', 'REA', 'SRF', 'PED'])
        for img, label in tqdm(zip(images_list, labels_list), total=len(labels_list)) :
            # print(label)
            mask = cv2.imread(label, 0)  # 0代表取灰度图

            s = np.unique(mask)  # 函数返回一个排好序的元素值唯一的array,意思是是去除重复的，然后排序输出
            # print(s)  # 返回[0 128 191 255]

            # 通道组合
            s_len = len(s)
            item = []
            if s_len == 1:
                item = [img, 0, 0, 0] # 正常样本
            elif s_len == 2:
                item = [img, 1, 0, 0]  # REA：1， SRF：0， PED：0
            elif s_len == 3:
                if s[1] == 128:
                    item = [img, 1, 0, 1]  # REA：1， SRF：0， PED：1
                elif s[1] == 191:
                    item = [img, 1, 1, 0]  # REA：1， SRF：1， PED：0
            elif s_len == 4:
                item = [img, 1, 1, 1]  # REA：1， SRF：1， PED：1

            list.append(item)

        # print(list)
        f = open(csv_dir, 'w', newline='')
        writer = csv.writer(f)
        writer.writerows(list)
        f.close()



if __name__ == '__main__':

    # # 生成并保存为 detection 的训练集 csv文件
    # data_to_csv(TRAIN_DATA_ROOT, mode='detection', csv_dir='train.csv')
    # data_to_csv(VAL_DATA_ROOT, mode='detection', csv_dir='val.csv')

    cube_img_label_list = get_cube_list(TRAIN_DATA_ROOT, have_label=True)
    print(cube_img_label_list[0][0])
    print(cube_img_label_list[0][1])
    file_list = get_file_list(cube_img_label_list[0][0], cube_img_label_list[0][1])


    img_cube = []
    mask_cube = []

    for file_path in file_list:

        img = cv2.imread(file_path[0], 0)
        for i in range(2):
            img = cv2.pyrDown(img)  # 图像金字塔下采样
        img_cube.append(img)
        cv2.imshow('img', img)

        mask = cv2.imread(file_path[1], 0)
        for i in range(2):
            mask = cv2.pyrDown(mask)  # 图像金字塔下采样
        mask_cube.append(mask)
        cv2.imshow('mask', mask)

        cv2.waitKey(0)

    img_cube = np.asarray(img_cube, np.uint8)
    mask_cube = np.asarray(mask_cube, np.uint8)
    np.save('img_cube.npy', img_cube)
    np.save('mask_cube.npy', mask_cube)
    print(img_cube.shape)
    print(mask_cube.shape)
