from fastai.vision import *
from fastai.vision.gan import *
from fastai import *
import os
import random
import time
import numpy as np
import torch
from torch import nn
from torch.backends import cudnn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F
from tqdm import tqdm, trange
from model.network import SSLI
from tool.dataloader import SsliDataset
from metric.loss import mask_loss
import gc
from sklearn.model_selection import train_test_split


# 划分数据集
all_file_paths = [os.path.join('./data', f) for f in os.listdir('data') if f.endswith('.csv')]
train_paths, test_paths = train_test_split(all_file_paths, test_size=0.2, random_state=42)
test_paths, val_paths = train_test_split(test_paths, test_size=0.5, random_state=42)

print(f"train_paths: {len(train_paths)}")
print(f"test_paths: {len(test_paths)}")
print(f"val_paths: {len(val_paths)}")
mean = [ 546.7953,  649.5146,  967.6898, 1159.2480, 2442.1875, 2433.1270, 1822.9164]
std =  [ 447.3463,  519.1241,  668.2700,  904.4800, 1166.9471, 1226.1171, 1154.9677]
# 创建数据集实例
train_dataset = SsliDataset(train_paths, mean, std, ratio=0.3)
val_dataset = SsliDataset(val_paths, mean, std, ratio=0.4)
test_dataset = SsliDataset(test_paths, mean, std)

ratio = 0.3
batch_size = 32
num_workers = 4

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=False,
    num_workers=num_workers)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=False,
    num_workers=num_workers)


# model = SSLI(7,256,4,8,7,0.2)

# train_data = ImageDataBunch(train_dl=train_loader, valid_dl=test_loader)
# train_data.sanity_check()
# save_name = 'SSLI'
# save_path = os.path.join('models', save_name)
# if not os.path.exists(save_path): os.mkdir(save_path)

# model = SSLI(7,256,4,8,7,0.2).cuda()
# #model = nn.DataParallel(model)
# learn = Learner(train_data, model, model_dir=save_path, loss_func=mask_loss())
# learn.fit_one_cycle(50,1e-4, callbacks=[
#     callbacks.SaveModelCallback(learn, every='improvement', monitor='valid_loss', name=f'best_val'),
#     callbacks.CSVLogger(learn, os.path.join(save_path, 'record'))], wd=1e-3)
# learn.save('model_49', with_opt=False)
