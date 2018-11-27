#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''

Author   : MG Studio
Datetime : 2018/11/16
Filename : Test_detections.py
'''




from Train_detections import *
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn import metrics
import numpy as np
from scipy import interp
# import matplotlib.pyplot as plt



# def getAuc(labels, pred) :
#     '''将pred数组的索引值按照pred[i]的大小正序排序，返回的sorted_pred是一个新的数组，
#        sorted_pred[0]就是pred[i]中值最小的i的值，对于这个例子，sorted_pred[0]=8
#     '''
#     sorted_pred = sorted(range(len(pred)), key = lambda i : pred[i])
#     pos = 0.0 #正样本个数
#     neg = 0.0 #负样本个数
#     auc = 0.0
#     last_pre = pred[sorted_pred[0]]
#     count = 0.0
#     pre_sum = 0.0  #当前位置之前的预测值相等的rank之和，rank是从1开始的，所以在下面的代码中就是i+1
#     pos_count = 0.0  #记录预测值相等的样本中标签是正的样本的个数
#     for i in range(len(sorted_pred)) :
#         if labels[sorted_pred[i]] > 0:
#             pos += 1
#         else:
#             neg += 1
#         if last_pre != pred[sorted_pred[i]]: #当前的预测概率值与前一个值不相同
#             #对于预测值相等的样本rank需要取平均值，并且对rank求和
#             auc += pos_count * pre_sum / count
#             count = 1
#             pre_sum = i + 1     #更新为当前的rank
#             last_pre = pred[sorted_pred[i]]
#             if labels[sorted_pred[i]] > 0:
#                 pos_count = 1   #如果当前样本是正样本 ，则置为1
#             else:
#                 pos_count = 0   #反之置为0
#         else:
#             pre_sum += i + 1    #记录rank的和
#             count += 1          #记录rank和对应的样本数，pre_sum / count就是平均值了
#             if labels[sorted_pred[i]] > 0:#如果是正样本
#                 pos_count += 1  #正样本数加1
#     auc += pos_count * pre_sum / count #加上最后一个预测值相同的样本组
#     auc -= pos *(pos + 1) / 2 #减去正样本在正样本之前的情况
#     auc = auc / (pos * neg)  #除以总的组合数
#     return auc

# 计算AUC
def getAuc(labels, pred) :
    fpr, tpr, thresholds = metrics.roc_curve(labels, pred, pos_label=1)
    # print(fpr, tpr, thresholds)
    auc = metrics.auc(fpr, tpr)
    # print(auc)
    return auc


def aic_fundus_lesion_classification(ground_truth, prediction, num_samples=128):
    """
    Classification task auc metrics.
    :param ground_truth: numpy matrix, (num_samples, 3)
    :param prediction: numpy matrix, (num_samples, 3)
    :param num_samples: int, default 128
    :return list:[AUC_1, AUC_2, AUC_3]
    """
    assert (ground_truth.shape == (num_samples, 3))
    assert (prediction.shape == (num_samples, 3))

    try:
        ret = [0.5, 0.5, 0.5]
        for i in range(3):
            fpr, tpr, thresholds = metrics.roc_curve(ground_truth[:, i], prediction[:, i], pos_label=1)
            ret[i] = metrics.auc(fpr, tpr)
    except Exception as e:

        print("ERROR msg:", e)
        return None
    return ret


# 预测单张图片
def pred_one_img_class(model, imagepath, use_gpu=True):
    img = cv2.imread(imagepath, 0)
    img = cv2.pyrDown(img)  # 图像金字塔下采样
    img = np.expand_dims(img, axis=2)
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[MEAN], std=[STD])   # 归一化
    ])
    img = trans(img)

    img = Variable(img.unsqueeze(0))
    if use_gpu: img = img.cuda()
    output = model(img)  # 模型预测
    output = torch.nn.Sigmoid()(output).data  # 经过一个Sigmoid函数

    # 概率值和预测类别
    P1, pred1 = torch.max(output[:, 0:2], 1)
    P2, pred2 = torch.max(output[:, 2:4], 1)
    P3, pred3 = torch.max(output[:, 4:6], 1)

    # return [pred1, pred2, pred3]

    # 输出为npy数组
    p1 = P1.cpu().numpy()[0] if use_gpu == True else P1.numpy()[0]
    p2 = P2.cpu().numpy()[0] if use_gpu == True else P2.numpy()[0]
    p3 = P3.cpu().numpy()[0] if use_gpu == True else P3.numpy()[0]

    if pred1 == 1:
        out1 = p1
    else:
        out1 = 1.0 - p1

    if pred2 == 1:
        out2 = p2
    else:
        out2 = 1.0 - p2

    if pred3 == 1:
        out3 = p3
    else:
        out3 = 1.0 - p3

    return [out1, out2, out3]







# 计算验证集AUC
def torch_model_val_auc(model):

    train_dataset = Datasets_detections(TRAIN_DATA_ROOT, TRAIN_CSV_DIR)
    print(train_dataset.__len__())

    val_dataset = Datasets_detections(VAL_DATA_ROOT, VAL_CSV_DIR)
    print(val_dataset.__len__())

    # 验证全体
    total_dataset = train_dataset + val_dataset
    total = len(total_dataset)
    print(total)

    # model = Net().cuda()
    # # print(model)
    #
    # # 模型加载
    # load_model(model, save_model_name)
    #
    # # 验证模式
    # model.eval()

    result = []
    labels = []

    for i in tqdm(range(total), total=total):
        img, label = total_dataset[i]
        img = Variable(img.unsqueeze(0)).cuda()
        output = model(img) #模型预测

        output = torch.nn.Sigmoid()(output).data
        # print(output)
        # print(output[:, 0:2])

        # 概率值和预测类别
        P1, pred1 = torch.max(output[:, 0:2], 1)
        P2, pred2 = torch.max(output[:, 2:4], 1)
        P3, pred3 = torch.max(output[:, 4:6], 1)

        label = label.unsqueeze(0) # 加一个维度
        # print(label)
        # print(label[:, 0:2])

        _, targ1 = torch.max(label[:, 0:2], 1)
        _, targ2 = torch.max(label[:, 2:4], 1)
        _, targ3 = torch.max(label[:, 4:6], 1)

        # print('targ1={}, pred1={}, P1={}'.format(targ1.cpu().numpy(), pred1.cpu().numpy(), P1.cpu().numpy()))
        # print('targ2={}, pred2={}, P2={}'.format(targ2.cpu().numpy(), pred2.cpu().numpy(), P2.cpu().numpy()))
        # print('targ3={}, pred3={}, P3={}'.format(targ3.cpu().numpy(), pred3.cpu().numpy(), P3.cpu().numpy()))

        # 输出为npy数组
        p1 = P1.cpu().numpy()[0]
        p2 = P2.cpu().numpy()[0]
        p3 = P3.cpu().numpy()[0]

        t1 = targ1.cpu().numpy()[0]
        t2 = targ2.cpu().numpy()[0]
        t3 = targ3.cpu().numpy()[0]

        if pred1 == 1:
            out1 = p1
        else:
            out1 = 1.0 - p1

        if pred2 == 1:
            out2 = p2
        else:
            out2 = 1.0 - p2

        if pred3 == 1:
            out3 = p3
        else:
            out3 = 1.0 - p3

        # if pred1 == 1:
        #     out1 = p1
        #
        #     if pred2 == 1:
        #         out2 = p2
        #     else:
        #         out2 = 1.0 - p2
        #
        #     if pred3 == 1:
        #         out3 = p3
        #     else:
        #         out3 = 1.0 - p3
        #
        # else:
        #     out1 = 1.0 - p1
        #     out2 = 0.0
        #     out3 = 0.0

        result.append([out1, out2, out3])
        labels.append([t1, t2, t3])


    result = np.asarray(result, np.float32)
    # result = np.round(result, decimals = 1) # 四舍五入取整

    labels = np.asarray(labels, np.float32)
    # labels = np.round(labels, decimals = 1)  # 四舍五入取整


    # print(result)
    # print(labels)
    #
    # print(result[:,2])
    # print(labels[:,2])

    # fpr, tpr, thresholds = metrics.roc_curve(labels[:,0], result[:,0], pos_label=1)
    # print(fpr, tpr, thresholds)
    # auc = metrics.auc(fpr, tpr)
    # print(auc)

    # # 计算AUC值
    # AUC1 = getAuc(labels[:, 0], result[:, 0])
    # AUC2 = getAuc(labels[:, 1], result[:, 1])
    # AUC3 = getAuc(labels[:, 2], result[:, 2])
    # AUC = (AUC1 + AUC2 + AUC3) / 3
    #
    # print('AUC_RED = %f' % AUC1)
    # print('AUC_SRF = %f' % AUC2)
    # print('AUC_PED = %f' % AUC3)
    # print('AUC_AVG = %f' % AUC)


    # # 计算roc曲线
    # fpr, tpr, thresholds = metrics.roc_curve(labels[:, 0], result[:, 0], pos_label=1)  # pos_label=1，表示值为1的实际值为正样本
    # print('fpr:', fpr)
    # print('tpr:', tpr)
    # print('thresholds:', thresholds)
    #
    # roc_auc = auc(fpr, tpr)
    # # 画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来
    # plt.plot(fpr, tpr, lw=1, label='ROC fold (area = %0.2f)' % roc_auc)
    # plt.show()

    res = aic_fundus_lesion_classification(labels, result, num_samples=total)
    print('auc: ', res)
    res_mean = (res[0] + res[1] + res[2]) / 3.0
    print('auc_mean:', res_mean)




if __name__ == '__main__':

    # 导入模型
    model = Net3().cuda()
    # print(model)
    # 权重加载
    save_model_name = 'detections_03'
    load_model(model, save_model_name)

    # 验证模式
    model.eval()

    # 测试所有样本AUC值
    torch_model_val_auc(model)
