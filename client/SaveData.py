import h5py
import numpy as np
from constVar import DOWNSAMPLE_SIZE,CHANEL_NUM

# 打开 HDF5 文件，如果文件不存在将被创建
def saveData(matrix, flag,path, name):
    fullpath = path + name
    with h5py.File(fullpath, 'a') as h5_file:
        # 检查数据集是否已经存在
        if 'eegdata' not in h5_file:
            # 如果数据集不存在，则创建数据集
            data_dset = h5_file.create_dataset('eegdata', (0, CHANEL_NUM, DOWNSAMPLE_SIZE*4), maxshape=(None, CHANEL_NUM, DOWNSAMPLE_SIZE*4), dtype='float64')
            labels_dset = h5_file.create_dataset('labels', (0,), maxshape=(None,), dtype='int')
        else:
            # 如果数据集已经存在，则获取数据集对象
            data_dset = h5_file['eegdata']
            labels_dset = h5_file['labels']
        
        np_matrix = np.array(matrix)
        
        # 追加样本到数据集
        new_data_shape = data_dset.shape[0] + 1
        data_dset.resize((new_data_shape, CHANEL_NUM, DOWNSAMPLE_SIZE*4))  # 调整数据集大小
        data_dset[-1] = np_matrix  # 在末尾追加新数据
        
        new_labels_shape = labels_dset.shape[0] + 1
        labels_dset.resize((new_labels_shape,))  # 调整数据集大小
        labels_dset[-1] = flag  # 在末尾追加新标签

def read_data(path):
    pass