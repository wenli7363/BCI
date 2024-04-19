import h5py
import numpy as np
def import_data(file_path):
    """
    导入h5py文件中的数据
    
    参数:
    file_path (str): h5py文件的路径
    
    返回:
    X (list): EEG数据,形状为[样本数, num_channels, num_timepoints]
    y (list): 标签数据,形状为[样本数]
    """
    X, y = [], []
    with h5py.File(file_path, 'r') as f:
        eegdata = f['eegdata']  # 假设EEG数据集的名称为'eegdata'
        labels = f['labels']    # 假设标签数据集的名称为'labels'
        
        num_subjects = eegdata.shape[0]     # 样本数
        for i in range(num_subjects):
            X.append(eegdata[i])            # EEG数据的列表
            y.append(labels[i])             # 标签列表
    
    # 如果需要进行标准化或其他预处理,可以在这里添加相关代码
    X_np = np.array(X)
    y_np = np.array(y)
    return X_np, y_np

# X,y = import_data("D:/Desktop/2/2_1.h5")
# print(X.shape)
# print(y.shape)