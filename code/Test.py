#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''

Author   : MG Studio
Datetime : 2018/11/16
Filename : Test.py
'''

from Train import *

Size_pad = 32

#------------------------------------------------------------------------------------------------
# 对图像边缘进行镜像扩展，使扩展后的图片长宽均为size的倍数
#-------------------------------------------------------------------------------------------------
def overlap_title(img, size=Size_pad):

    ih = img.shape[0]
    iw = img.shape[1]
    oh = (ih // size + int(ih % size > 0))*size
    ow = (iw // size + int(iw % size > 0))*size
    p_up = (oh - ih) // 2
    p_down = oh - ih - p_up
    p_left = (ow - iw) //2
    p_right = ow - iw - p_left
    pad_img = np.pad(img, ((p_up, p_down), (p_left, p_right), (0,0)), 'reflect')

    return pad_img, (p_up, p_up + ih, p_left, p_left + iw)

# 预测单张图片
def pred_one_img_mask(image, model, as_label=False, normal=True, use_gpu=True):
    # [1] 加载图像，提取前3个通道
    # img_ = io.imread(path)

    img_ = np.expand_dims(image, axis=2)

    img_, region = overlap_title(img_)

    # [2] 转化为channel-1st 的 4d tensor
    # img_ = np.expand_dims(image, axis=2)
    img = transforms.ToTensor()(img_)  # 图像转为tensor
    if normal:
        img = transforms.Normalize(mean=[MEAN], std=[STD])(img)  # 归一化
        img = Variable(img.unsqueeze(0))
    if use_gpu:
        img = img.cuda()

        print(img.shape)

    # [3] 输出预测概率图，并阈值化
    output = model(img)

    output_sig = nn.Sigmoid()(output)  # 经过一个sigmoid

    # [4] 转化为2-d array， 并crop出和输出一致的区域

    if as_label:
        P, pred = torch.max(output_sig, dim=1)
        pred_msk = pred.data.squeeze().cpu().numpy() if use_gpu==True \
            else pred.data.squeeze().numpy()
        pred_msk = pred_msk[region[0]:region[1], region[2]:region[3]]

    else:
        pred_msk = output_sig.data.squeeze().cpu().numpy().transpose(1, 2, 0) if use_gpu==True \
            else output_sig.data.squeeze().numpy().transpose(1, 2, 0)
        pred_msk = pred_msk[region[0]:region[1], region[2]:region[3], :]

    print("pred_msk", pred_msk.shape)

    return pred_msk.astype(np.uint8)


# 标记对比
def pred_to_mask(pred_msk, th1=0.6, th2=0.7, th3=0.7):
    pred = np.zeros((512, 256), np.uint8)
    pred[pred_msk[:, :, 1] > th1] = 1
    pred[pred_msk[:, :, 2] > th2] = 2
    pred[pred_msk[:, :, 3] > th3] = 3

    plt.imshow(pred, cmap='gray')
    plt.show()

    return pred


# AIC眼底病变分割， 计算dice系数
def aic_fundus_lesion_segmentation(ground_truth, prediction, num_samples=128):
    """
    Detection task auc metrics.
    :param ground_truth: numpy matrix, (num_samples, 1024, 512)
    :param prediction: numpy matrix, (num_samples, 1024, 512)
    :param num_samples: int, default 128
    :return list:[Dice_0, Dice_1, Dice_2, Dice_3]
    """
    assert (ground_truth.shape == (num_samples, 1024, 512))
    assert (prediction.shape == (num_samples, 1024, 512))

    ground_truth = ground_truth.flatten()
    prediction = prediction.flatten()
    try:
        ret = [0.0, 0.0, 0.0]
        for i in range(4):
            mask1 = (ground_truth == i)
            mask2 = (prediction == i)
            if mask1.sum() != 0:
                ret[i] = 2 * ((mask1 * (ground_truth == prediction)).sum()) / (mask1.sum() + mask2.sum())
            else:
                ret[i] = float('nan')
    except Exception as e:
        print("ERROR msg:", e)
        return None

    return ret




if __name__ == "__main__":


    # 加载模型
    unet = Unet(1, 4).cuda()

    # 模型加载训练权重
    save_model_name = 'unet1'
    load_model(unet, save_model_name)

    # 验证模式
    unet = unet.eval()

    # 数据集加载
    train_dataset = Datasets(TRAIN_DATA_ROOT, s_trans, t_trans)
    print(len(train_dataset))

    # 测试集加载
    val_dataset = Datasets(VAL_DATA_ROOT, s_trans, t_trans)
    print(len(val_dataset))

    # 验证集全体数据
    total_dataset = train_dataset + val_dataset
    total = len(total_dataset)
    print(total)

    labels = []
    preds = []
    for i in tqdm(range(total), total=total):
    # for i in range (0, 128):
        img, mask = total_dataset[i]
        img = img.squeeze().numpy()
        mask = mask.numpy().astype(np.uint8)
        print(img.shape)
        print(mask.shape)

        pred1 = pred_one_img_mask(img, unet, as_label=False)  # 预测结果
        pred1_ = pred_to_mask(pred1)  #

        pred2 = pred_one_img_mask(img, unet, as_label=True, normal=False)  # 预测结果
        print(pred2.shape)  #

        # 显示结果
        plt.subplot(241)
        plt.title('img')
        plt.imshow(img, cmap='gray')
        plt.subplot(242)
        plt.title('true mask')
        plt.imshow(mask, cmap='gray')
        plt.subplot(243)
        plt.title('predict mask')
        plt.imshow(pred2, cmap='gray')
        plt.subplot(244)
        plt.title('predict mask_')
        plt.imshow(pred1_, cmap='gray')
        plt.subplot(245)
        plt.title('predict mask0')
        plt.imshow(pred1[:, :, 0], cmap='gray')
        plt.subplot(246)
        plt.title('predict mask1')
        plt.imshow(pred1[:, :, 1], cmap='gray')
        plt.subplot(247)
        plt.title('predict mask2')
        plt.imshow(pred1[:, :, 2], cmap='gray')
        plt.subplot(248)
        plt.title('predict mask3')
        plt.imshow(pred1[:, :, 3], cmap='gray')
        plt.show()

        result = pred_one_img_mask(img, unet, as_label=False, normal=False)  # 预测结果
        result = pred_to_mask(result)

        label = cv2.pyrUp(mask)  # 上采样
        pred = cv2.pyrUp(result)  # 上采样
        plt.imshow(label, cmap='gray')
        plt.imshow(pred, cmap='gray')
        plt.show()

        print(label.shape)
        print(pred.shape)

        labels.append(label)
        preds.append(pred)

    labels = np.asarray(labels, np.uint8)
    preds = np.asarray(preds, np.uint8)
    print(labels.shape)
    print(preds.shape)

    dice = aic_fundus_lesion_segmentation(labels, preds, num_samples=total)  # 计算 dice 系数
    print(dice)
