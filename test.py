from tool.dataloader import SsliDataset
import os
import numpy as np
all_file_paths = [os.path.join('./data', f) for f in os.listdir('data') if f.endswith('.csv')]
print(all_file_paths)
mean = np.array([5,5,5,5,5,5,5])
std = np.array([3,3,3,3,3,3,3])
dataset = SsliDataset(all_file_paths, mean, std, 0.3)
print(dataset[0][1].shape)