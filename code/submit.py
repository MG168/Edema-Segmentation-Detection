#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''

Author   : MG Studio
Datetime : 2018/11/16
Filename : submit.py
'''


import numpy as np
import matplotlib.pyplot as plt
from Test_detections import *
from Test import *


root_path = '../data/Edema_validationset/groundtruth/'
detections_file = 'P0067_MacularCube512x128_4-28-2014_17-15-21_OS_sn21294_cube_z_labelMark_detections.npy'
volumes_file = 'P0067_MacularCube512x128_4-28-2014_17-15-21_OS_sn21294_cube_z_labelMark_volumes.npy'


submit_path = '../data/submission_example/'
s_detections_file = 'P0128_MacularCube512x128_6-27-2013_9-30-22_OD_sn5723_cube_z_detections.npy'
s_volumes_file = 'P0128_MacularCube512x128_6-27-2013_9-30-22_OD_sn5723_cube_z_volumes.npy'
save_path = '../data/results/'

# 创建提交目录
if(not (os.path.exists(submit_path))):
    os.mkdir(submit_path)

if(not (os.path.exists(save_path))):
    os.mkdir(save_path)


# 加载detections_file
def load_detections(file):
    detections = np.load(file)
    print(detections.shape)
    print(detections)
    return detections


# 加载volumes_file
def load_volumes(file):
    volumes = np.load(file)
    print(volumes.shape)
    print(volumes)
    return volumes


# 预测一个cube
def test_on_cube(model, testcube_path, submit_path, mode='detections'):

    if mode == 'detections':

        results = []

        # 获取cube文件的图片
        file_list = get_file_list(testcube_path, None)

        for file_path in tqdm(file_list, total=len(file_list)):
            # print(file_path)
            result = pred_one_img_class(model, file_path)
            results.append(result)

        results = np.asarray(results, np.float32)
        # results = np.round(results, decimals=1)  # 四舍五入取整
        # print(results)

        npy_name = testcube_path.split('/')[-2].split('.')[0] + '_detections.npy'
        np.save(submit_path + npy_name, results)

    elif mode == 'volumes':

        results = []

        # 获取cube文件的图片
        file_list = get_file_list(testcube_path, None)

        for file_path in tqdm(file_list, total=len(file_list)):
            img = cv2.imread(file_path, 0)
            img = cv2.pyrDown(img)  # 图像金字塔下采样

            result = pred_one_img_mask(img, model, as_label=True, normal=True) # 预测结果
            # result = pred_one_img_mask(img, model, as_label=False, normal=True)
            # result = pred_to_mask(result, th1=0.6, th2=0.7, th3=0.7)

            # plt.imshow(result, cmap='gray')
            # plt.show()

            result = cv2.pyrUp(result)

            results.append(result)

        results = np.asarray(results, np.uint8)
        # print(results.shape)

        npy_name = testcube_path.split('/')[-2].split('.')[0] + '_volumes.npy'
        np.save(submit_path + npy_name, results)


def show_result():
    detections = load_detections(submit_path + s_detections_file)
    volumes = load_volumes(submit_path + s_volumes_file)
    for i in range(128):
        print(volumes[i].max())
        plt.imshow(volumes[i], cmap='gray')
        plt.savefig(save_path + '{}.png'.format(i))
        # plt.show()


def submit_result():

    #############################生成提交文件######################################################
    # 导入模型
    model = Net3().cuda()
    # print(model)
    # 权重加载
    load_model(model, 'detections_03')
    # 验证模式
    model.eval()

    # # # 单个cube
    # cube_img_list = get_cube_list(TEST_DATA_ROOT, have_label=False)  # 获取测试文件列表
    # test_on_cube(model, cube_img_list[0], './', mode='detections')  # 测试第一个cube
    # test_on_cube(model, cube_img_list[0], './', mode='volumes')

    cube_img_list = get_cube_list(TEST_DATA_ROOT, have_label=False)
    for cube in cube_img_list:
        test_on_cube(model, cube, submit_path=submit_path, mode='detections')


    # 加载模型
    unet = Unet(1, 4).cuda()

    # 模型加载预训练权重
    load_model(unet, 'unet1')

    # 验证模式
    unet = unet.eval()

    cube_img_list = get_cube_list(TEST_DATA_ROOT, have_label=False)
    for cube in cube_img_list:
        test_on_cube(unet, cube, submit_path=submit_path, mode='volumes')





if __name__ == '__main__':
##################测试代码###########################################
    submit_result()
    # show_result()

    # detections = load_detections('PC002_MacularCube512x128_8-28-2013_9-1-44_OD_sn8650_cube_z_detections.npy')
    # print(detections)

    # volumes = load_volumes('PC002_MacularCube512x128_8-28-2013_9-1-44_OD_sn8650_cube_z_volumes.npy')
    # print(volumes.shape)

    # list_file = os.listdir(submit_path)
    # print(list_file)
    # detections = load_volumes(submit_path + list_file[0])
    # print(detections.shape)
