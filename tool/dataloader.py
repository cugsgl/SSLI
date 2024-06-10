import torch.utils.data as data
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
    
class SsliDataset(data.Dataset):
    def __init__(self, file_paths, mean, std, ratio=None):
        self.file_paths = file_paths
        self.ratio = ratio
        self.file_index = []
        self.lst = []
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.head = []
        for i in range(122):
            self.lst.append(f'{i:03d}.coastal_blue')
            self.lst.append(f'{i:03d}.blue')
            self.lst.append(f'{i:03d}.green')
            self.lst.append(f'{i:03d}.red')
            self.lst.append(f'{i:03d}.nir')
            self.lst.append(f'{i:03d}.swir1')
            self.lst.append(f'{i:03d}.swir2')
            self.lst.append(f'{i:03d}.qa')
        # 创建文件索引
        for file_path in tqdm(self.file_paths):
            data_part = pd.read_csv(file_path)
            self.file_index.extend([(file_path, idx) for idx in range(1,len(data_part)+1)])
            if len(self.head) == 0 :
                self.head = pd.read_csv(file_path, nrows=0).columns.tolist()

    def __len__(self):
        return len(self.file_index)

    def __getitem__(self, idx):
        file_path, row_idx = self.file_index[idx]
        # 读取对应文件和idx的数据
        data_part = pd.read_csv(file_path, skiprows=row_idx, nrows=1, header=None)
        data_part.columns = self.head
        return self.transform(data_part[self.lst], self.mean, self.std)
    
    def transform(self, raw_data, mean, std):
        # 归一化
        mat_data = raw_data[self.lst].values.reshape(122,8)
        mat_data = mat_data.astype(np.float64)

        clo = np.where(mat_data[:,7]==1)
        mat_data[clo,:7] = (mat_data[clo,:7] - mean) / std

        ts = np.zeros((122, 8))
        ts[:,:7] = mat_data[:,:7].astype(np.float64) # TC
        qa = mat_data[:,7].astype(np.float64)
        if self.ratio is not None:
            clr_inx = np.argwhere(qa!=0)[:,0]
            clr_num = int(clr_inx.shape[0] * self.ratio)
            clr_num = clr_num if clr_num >= 1 else 1
            mask_inx = np.random.choice(clr_inx, size=clr_num, replace=False)
        else: 
            mask_inx  = np.argwhere(qa==0)[:,0]
        # 掩膜的数据设置为0
        ts[mask_inx,:7] = 0
        qa[mask_inx] = 0
        # 无效值
        ts[:,:7][qa==0] = 0
        ts[:,-1][qa==1] = 1
        ts = torch.from_numpy(ts).float()
        gt = torch.from_numpy(mat_data[:,:7]).float()
        mask = torch.zeros_like(gt)
        mask[mask_inx] = 1
        return ts, torch.cat((gt,mask),dim=1)