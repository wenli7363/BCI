import h5py
import numpy as np
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# 用于导入mat数据，并转为numpy数组格式
# every: 是否导入所有电极
# number: 受试者数据编号
# 返回处理好的训练集X(9,288,25,1000)，标签y(288)
# 用于导入mat数据，并转为numpy数组格式
# every: 是否导入所有电极
# number: 受试者数据编号
# 返回处理好的训练集X(9,288,25,1000)，标签y(288)
def import_data(every=False,number=1):
    # 是否引入所有电极
    # 是否引入所有电极
    if every:
        electrodes = 25
    else:
        electrodes = 22
    X, y = [], []
    for i in range(9):
        # if(i==number):continue
    
        print('/root/autodl-tmp/data/A0' + str(i + 1) + 'T_slice.mat')
        A01T = h5py.File('./data/A0' + str(i + 1) + 'T_slice.mat', 'r')
        X1 = np.copy(A01T['image'])
        X.append(X1[:, :electrodes, :])     # 注意是append进去的，所以升了一个维度，第一维是受试者编号
        y1 = np.copy(A01T['type'])
        y1 = y1[0, 0:X1.shape[0]:1]
        y.append(np.asarray(y1, dtype=np.int32))    # 转成numpy数组

    for subject in range(9):
        # if(i==number):continue
        delete_list = []
        for trial in range(288):
            # 如果存在一个NaN的数据，说明这次trail有问题，删掉
            if np.isnan(X[subject][trial, :, :]).sum() > 0:
                delete_list.append(trial)
        X[subject] = np.delete(X[subject], delete_list, 0)
        y[subject] = np.delete(y[subject], delete_list)
    y = [y[i] - np.min(y[i]) for i in range(len(y))]    # 标签做个映射，从0开始
    return X, y




def import_data_test(every=False):
    if every:
        electrodes = 25
    else:
        electrodes = 22
    X, y = [],[]
    for i in range(9):
        B01T = h5py.File('grazdata/B0' + str(i + 1) + 'T.mat', 'r')
        X1 = np.copy(B01T['image'])
        X.append(X1[:, :electrodes, :])
        y1 = np.copy(B01T['type'])
        y1 = y1[0, 0:X1.shape[0]:1]
        y.append(np.asarray(y1, dtype=np.int32))

    for subject in range(9):
        delete_list = []
        for trial in range(288):
            if np.isnan(X[subject][trial, :, :]).sum() > 0:
                delete_list.append(trial)
        X[subject] = np.delete(X[subject], delete_list, 0)
        y[subject] = np.delete(y[subject], delete_list)
    y = [y[i] - np.min(y[i]) for i in range(len(y))]
    return X, y

def train_test_subject(X, y, train_all=True, standardize=True):

    l = np.random.permutation(len(X[0]))
    X_test = X[0][l[:50], :, :]
    y_test = y[0][l[:50]]

    if train_all:
        X_train = np.concatenate((X[0][l[50:], :, :], X[1], X[2], X[3], X[4], X[5], X[6], X[7], X[8]))
        y_train = np.concatenate((y[0][l[50:]], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8]))

    else:
        X_train = X[0][l[50:], :, :]
        y_train = y[0][l[50:]]

    X_train_mean = X_train.mean(0)
    X_train_var = np.sqrt(X_train.var(0))

    if standardize:
        X_train -= X_train_mean
        X_train /= X_train_var
        X_test -= X_train_mean
        X_test /= X_train_var

    X_train = np.transpose(X_train, (0, 2, 1))
    X_test = np.transpose(X_test, (0, 2, 1))

    return X_train, X_test, y_train, y_test


def train_test_total(X, y, standardize=True):

    X_total = np.concatenate((X[0], X[1], X[2], X[3], X[4], X[5], X[6], X[7], X[8]))
    y_total = np.concatenate((y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8]))

    l = np.random.permutation(len(X_total))
    X_test = X_total[l[:200], :, :]
    y_test = y_total[l[:200]]
    X_train = X_total[l[200:], :, :]
    y_train = y_total[l[200:]]

    X_train_mean = X_train.mean(0)
    X_train_var = np.sqrt(X_train.var(0))

    if standardize:
        X_train -= X_train_mean
        X_train /= X_train_var
        X_test -= X_train_mean
        X_test /= X_train_var

    X_train = np.transpose(X_train, (0, 2, 1))
    X_test = np.transpose(X_test, (0, 2, 1))

    return X_train, X_test, y_train, y_test    