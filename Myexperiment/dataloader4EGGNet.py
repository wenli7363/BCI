import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

def dataloader(X, y, batch_size=128, shuffle=True, num_workers=0, val_ratio=0.2):
    """
    将数据转换为PyTorch的DataLoader对象

    参数:
    X (list): EEG数据,形状为[num_subjects, num_channels, num_timepoints]
    y (list): 标签数据,形状为[num_subjects]
    batch_size (int): 批量大小
    shuffle (bool): 是否对数据进行随机打乱
    num_workers (int): 使用多少个子进程加载数据
    val_ratio (float): 验证集占总数据的比例

    返回:
    train_loader (DataLoader): 训练数据的DataLoader对象
    val_loader (DataLoader): 验证数据的DataLoader对象
    test_loader (DataLoader): 测试数据的DataLoader对象
    """
    # 增加一个维度
    X = np.expand_dims(X, axis=1)   
    # 将列表数据转换为PyTorch的Tensor
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    # 创建TensorDataset
    dataset = TensorDataset(X_tensor, y_tensor)

    # 划分训练集、验证集和测试集
    train_val_size = int((1 - val_ratio) * len(dataset))    # 训练集和验证集的总长度
    test_size = len(dataset) - train_val_size               # 测试集的长度
    train_val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_val_size, test_size])

    # 划分训练集和验证集
    train_size = int(train_val_size * (1 - val_ratio))
    val_size = train_val_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [train_size, val_size])

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader