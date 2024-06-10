import os
import pandas as pd
from tqdm import tqdm, trange
import numpy as np
import torch

directory_path = './data'

# 定义列名合并映射规则
column_merge_mapping = {}
lst = []
for i in range(122):
    column_merge_mapping[f'{i:03d}.coastal_blue'] = 'coastal_blue'
    column_merge_mapping[f'{i:03d}.blue'] = 'blue'
    column_merge_mapping[f'{i:03d}.green'] = 'green'
    column_merge_mapping[f'{i:03d}.red'] = 'red'
    column_merge_mapping[f'{i:03d}.nir'] = 'nir'
    column_merge_mapping[f'{i:03d}.swir1'] = 'swir1'
    column_merge_mapping[f'{i:03d}.swir2'] = 'swir2'
    column_merge_mapping[f'{i:03d}.qa'] = 'qa'
    lst.append(f'{i:03d}.coastal_blue')
    lst.append(f'{i:03d}.blue')
    lst.append(f'{i:03d}.green')
    lst.append(f'{i:03d}.red')
    lst.append(f'{i:03d}.nir')
    lst.append(f'{i:03d}.swir1')
    lst.append(f'{i:03d}.swir2')
    lst.append(f'{i:03d}.qa')

# 初始化总和和计数
sum_values = None
sum_squares = None
count_values = None

# 遍历目录中的所有CSV文件
for file_name in tqdm(os.listdir(directory_path)):
    if file_name.endswith('.csv'):
        file_path = os.path.join(directory_path, file_name)
        print(f"path: {file_path}")
        # 读取CSV文件
        df = pd.read_csv(file_path, usecols=lst) 
        # 重命名列以合并

        train_mat_2021 = df.values.reshape(-1,122,8)
        train_mat = train_mat_2021.astype(np.float64)
        train_doy = train_mat[:,:,7].sum(axis=-1)
        train_mat = train_mat[train_doy>=2]
        raw_train = torch.from_numpy(train_mat[train_mat[:,:,7]==1][:,:7]).float()

        print
        # 累计总和
        if sum_values is None:
            sum_values = torch.sum(raw_train, dim=0, keepdim=True)
            sum_squares = torch.sum(raw_train ** 2, dim = 0, keepdim=True)
        else:
            sum_values += torch.sum(raw_train, dim=0, keepdim=True)
            sum_squares += torch.sum(raw_train ** 2, dim = 0, keepdim=True)
        # 累计计数
        if count_values is None:
            count_values = raw_train.shape[0]
        else:
            count_values +=  raw_train.shape[0]
        gc.collect()
        print(f"count_values: {count_values}")

# 计算均值
mean_values = sum_values / count_values

# 计算方差
variance = (sum_squares / count_values) - (mean_values ** 2)
# 计算标准差
std_dev = variance ** 0.5
print("Column Means:")
print(mean_values) 
# [ 546.7953,  649.5146,  967.6898, 1159.2480, 2442.1875, 2433.1270, 1822.9164]
print("Column std Variances:")
print(std_dev)
# [ 447.3463,  519.1241,  668.2700,  904.4800, 1166.9471, 1226.1171, 1154.9677]
