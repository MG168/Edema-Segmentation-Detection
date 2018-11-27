#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''

Author   : MG Studio
Datetime : 2018/11/16
Filename : Test_volumns.py
'''



from Train_volumes import *






if __name__ == '__main__':

    # 加载模型
    unet = Unet_3D(in_dim=1, out_dim=4, num_filter=8)

    # 模型加载
    load_model(unet, save_model_name)

    # 验证模式
    unet = unet.eval()

    # 验证集加载
    val_dataset_3d = Datasets_3D(VAL_DATA_ROOT)
    print(len(val_dataset_3d))


    cube_image, cube_label = val_dataset_3d[10]  # 获取第一个cube
    image = cube_image.squeeze(0).numpy()
    print(image.shape)
    np.save('out_image.npy', image)
    label = cube_label.squeeze(0).numpy()
    print(label.shape)
    np.save('out_label.npy', label)


    cube_image = cube_image.unsqueeze(0)
    cube_image = Variable(cube_image).cuda()

    print('insize:', cube_image.size())

    # 模型预测
    unet = unet.cuda()
    output = unet(cube_image)
    print('outsize:', output.size())

    out = pred_cube(output)  # 预测结果输出
    print(out.shape)
    np.save('out.npy', out)  # 保存预测结果
