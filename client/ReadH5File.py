import h5py

def read_h5_file(file_path):
    # 打开 HDF5 文件
    with h5py.File(file_path, 'r') as h5_file:
        # 打印 HDF5 文件中的所有对象（数据集和组）
        def print_objects(name, obj):
            if isinstance(obj, h5py.Dataset):
                # 如果对象是数据集，打印其名称和形状
                print(f"Dataset: {name}, Shape: {obj.shape}")
                # 打印数据集的前几个元素作为示例
                print(f"Data (first few elements): {obj[:5]}")
            elif isinstance(obj, h5py.Group):
                # 如果对象是组，打印其名称
                print(f"Group: {name}")

        # 遍历文件中的所有对象并打印其内容
        h5_file.visititems(print_objects)

# 使用示例
# file_path = 'D:\\Desktop\\2\\test2.h5'  # 替换为您的 HDF5 文件路径
# file_path = 'D:\\Desktop\\2\\2_1.h5'  # 替换为您的 HDF5 文件路径
# file_path = 'D:\\Desktop\\4\\42.h5'
file_path = 'D:\\Desktop\\2\\25.h5'
read_h5_file(file_path)
