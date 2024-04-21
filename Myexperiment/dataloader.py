import torch
from torch.utils.data import TensorDataset, DataLoader
from preprocessing import import_data
import numpy as np

# 将一个已存在的X, y数据集转换为PyTorch的DataLoader对象
def dataloader(X, y, batch_size=128, shuffle=True, num_workers=0):
    """
    将数据转换为PyTorch的DataLoader对象

    参数:
    X (list): EEG数据,形状为[num_subjects, num_channels, num_timepoints]
    y (list): 标签数据,形状为[num_subjects]
    batch_size (int): 批量大小
    shuffle (bool): 是否对数据进行随机打乱
    num_workers (int): 使用多少个子进程加载数据

    返回:
    train_loader (DataLoader): 训练数据的DataLoader对象
    test_loader (DataLoader): 测试数据的DataLoader对象
    """
    # 增加一个维度
    X = np.expand_dims(X, axis=1)   
    # 将列表数据转换为PyTorch的Tensor
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    # 创建TensorDataset
    dataset = TensorDataset(X_tensor, y_tensor)

    # 划分训练集和测试集
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return train_loader, test_loader

def append_data(loader, X, y):
    """
    向已有的DataLoader追加新数据

    参数:
    loader (DataLoader): 要追加数据的DataLoader对象
    X (list): 新的EEG数据,形状为[num_subjects, num_channels, num_timepoints]
    y (list): 新的标签数据,形状为[num_subjects]

    返回:
    new_loader (DataLoader): 包含原始数据和新数据的DataLoader对象
    """
    # 增加一个维度
    X = np.expand_dims(X, axis=1)
    # 将列表数据转换为PyTorch的Tensor
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    # 创建TensorDataset
    new_dataset = TensorDataset(X_tensor, y_tensor)

    # 合并原始数据集和新数据集
    combined_dataset = torch.utils.data.ConcatDataset([loader.dataset, new_dataset])

    # 创建新的DataLoader
    new_loader = DataLoader(combined_dataset, batch_size=loader.batch_size, shuffle=loader.shuffle, num_workers=loader.num_workers)

    return new_loader

# X, y = import_data('D:\\Desktop\\2\\25.h5')     # shape: (样本数, num_channels, num_timepoints)
# train_loader, test_loader = dataloader(X, y)

# data_zip = zip(train_loader, test_loader)
# for batch_idx, ((inputs_s, labels_s), (inputs_t, labels_t)) in enumerate(data_zip):

#     print(f"inputs_s shape: {inputs_s.shape}") 
#     print(f"labels_s shape: {labels_s.shape}")
#     print(f"inputs_t shape: {inputs_t.shape}")
#     print(f"labels_t shape: {labels_t.shape}")


# for batch in train_loader:
#     data,  targets = batch
#     print(data.shape)
#     print(targets.shape)

# for batch in test_loader:
#     data,  targets = batch
#     print(data.shape)
#     print(targets.shape)

