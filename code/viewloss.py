#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
绘制训练loss曲线和保存loss数据

Author   : MG Studio
Datetime : 2018/11/16
Filename : viewloss.py
'''



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# loss可视化
class ViewLoss():

    def __init__(self, labels):
        self.labels = labels # 标签
        self.losses_hist = [] # loss数据列表
        for _ in range(len(self.labels)):
            self.losses_hist.append([])

    # 绘制loss曲线
    def show_loss(self, xlabel='Steps', ylabel='Loss', title='train loss'):
        for i, l_his in enumerate(self.losses_hist):
            plt.plot(l_his, label=self.labels[i]) # 绘制曲线

        # 绘制参数设置
        plt.legend(loc='best')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.ylim((0, 1.0))
        plt.savefig('{}.png'.format(title))
        plt.show()

    # 保存loss参数
    def save_loss(self, csv='./loss.csv'):
        losses_hist = np.asarray(self.losses_hist).T # 转成numpy在转置
        loss_data = pd.DataFrame(columns=self.labels, data=losses_hist) #加载到panda里面
        loss_data.to_csv(csv) #保存成csv格式


    # 将csv绘制成曲线图
    def plot_loss(self, csv='./loss.csv', xlabel='Steps', ylabel='Loss', title='train loss'):
        data = pd.read_csv(csv) # 读取csv文件
        data = np.array(data).T[1:]

        for i, l_his in enumerate(data):
            plt.plot(l_his, label=self.labels[i]) # 绘制曲线

        # 绘制参数设置
        plt.legend(loc='best')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.ylim((0, 1.0))
        plt.savefig('{}.png'.format(title)) # 保存结果
        plt.show()



if __name__ == '__main__':


    # 实例化view
    labels = ['train_loss']
    view = ViewLoss(labels=labels)

    # for i in range(100):
    #     view.losses_hist[0].append(i)
    #     view.losses_hist[1].append(i * 0.1)
    #     # view.losses_hist[2].append(i * 0.01)
    #     # view.losses_hist[3].append(i * 0.001)
    # # 保存训练loss
    # view.save_loss(csv='./loss.csv')

    # 绘制训练曲线
    # view.plot_loss(csv='loss.csv')


    view.plot_loss(csv='volumes_train.csv')