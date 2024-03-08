import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import importlib 
import preprocessing
importlib.reload(preprocessing)
from preprocessing import *
import warnings
warnings.filterwarnings('ignore')
import torch
import pandas as pd
from scipy.io import loadmat
import  numpy as np
from sklearn.model_selection import train_test_split
import  torch.nn as nn

# 加载数据集并返回训练集和测试集的数据加载器对象。
# number是受试者编号
# 返回数据集加载器对象
def dataloader_2a(number=1):
    X, y = import_data(False,1)
    X_train,X_test,y_train,y_test = get_dataset_subject(X, y,number)

    # 多余的转置操作
    # X_train=np.transpose(X_train, [0,2,1]) 
    # X_test=np.transpose(X_test, [0,2,1]) 
    
    #########
    # 这里我要处理一下，把他们设置为等长度的，这样用的数据就多了
    # X_test=np.tile(X_test,(8,1,1))
    # print(y_test.shape)
    # y_test=np.tile(y_test,(8))
    # print(X_test.shape)
    # print(y_test.shape)
    # #######
    #######

    # 多增加一个维度，表示batch
    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)     

    X_train = torch.FloatTensor(X_train)
    X_test  = torch.FloatTensor(X_test)

    Y_train=torch.LongTensor(y_train)
    Y_test=torch.LongTensor(y_test)
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=True)



    return train_loader,test_loader 




def get_dataset_subject(X,y,number=1,standardize=True):
    # 除了number的那一列样本
    # 这个条件分支对应着number等于1时的情况，
    # 也就是选择除了第一个被试者之外的所有数据作为训练集，而第一个被试者的数据作为测试集。
    if(number==1):
        X_total = np.concatenate((X[1], X[2], X[3], X[4], X[5], X[6], X[7], X[8]))
        y_total = np.concatenate((y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8]))
    elif(number==2):
        X_total = np.concatenate((X[0], X[8], X[2], X[3], X[4], X[5], X[6], X[7]))
        y_total = np.concatenate((y[0], y[8], y[2], y[3], y[4], y[5], y[6], y[7]))
    elif(number==3):
        X_total = np.concatenate((X[0], X[1], X[8], X[3], X[4], X[5], X[6], X[7]))
        y_total = np.concatenate((y[0], y[1], y[8], y[3], y[4], y[5], y[6], y[7]))
    elif(number==4):
        X_total = np.concatenate((X[0], X[1], X[2], X[8], X[4], X[5], X[6], X[7]))
        y_total = np.concatenate((y[0], y[1], y[2], y[8], y[4], y[5], y[6], y[7]))
    elif(number==5):
        X_total = np.concatenate((X[0], X[1], X[2], X[3], X[8], X[5], X[6], X[7]))
        y_total = np.concatenate((y[0], y[1], y[2], y[3], y[8], y[5], y[6], y[7]))
    elif(number==6):
        X_total = np.concatenate((X[0], X[1], X[2], X[3], X[4], X[8], X[6], X[7]))
        y_total = np.concatenate((y[0], y[1], y[2], y[3], y[4], y[8], y[6], y[7]))
    elif(number==7):
        X_total = np.concatenate((X[0], X[1], X[2], X[3], X[4], X[5], X[8], X[7]))
        y_total = np.concatenate((y[0], y[1], y[2], y[3], y[4], y[5], y[8], y[7]))
    elif(number==8):
        X_total = np.concatenate((X[0], X[1], X[2], X[3], X[4], X[5], X[6], X[8]))
        y_total = np.concatenate((y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[8]))
    else:
        X_total = np.concatenate((X[0], X[1], X[2], X[3], X[4], X[5], X[6], X[7]))
        y_total = np.concatenate((y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]))

    # 测试集
    X_test=X[number-1]
    y_test=y[number-1]
    
    # 计算均值和标准差
    X_train_mean = X_total.mean(0)
    X_train_var = np.sqrt(X_total.var(0))

    X_test_mean = X_test.mean(0)
    X_test_var = np.sqrt(X_test.var(0))

    # 标准化
    if standardize:
        X_total -= X_train_mean
        X_total /= X_train_var
        X_test -= X_test_mean
        X_test /= X_test_var
       
    # 三个维度重新排列，改成了(样本数，特征数，通道数)
    # X_total = np.transpose(X_total, (0, 2, 1))
    # X_test = np.transpose(X_test, (0, 2, 1))
    
    return X_total,X_test,y_total,y_test