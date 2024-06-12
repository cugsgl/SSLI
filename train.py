import os
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from model.network import SSLI
from tool.dataloader import SsliDataset
from metric.loss import mask_loss
import gc
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from model.pl_model  import LitModel
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


# 划分数据集
all_file_paths = [os.path.join('./data', f) for f in os.listdir('data') if f.endswith('.csv')]
train_paths, test_paths = train_test_split(all_file_paths, test_size=0.2, random_state=42)
test_paths, val_paths = train_test_split(test_paths, test_size=0.5, random_state=42)

print(f"train_paths size: {len(train_paths)}")
print(f"test_paths size: {len(test_paths)}")
print(f"val_paths size: {len(val_paths)}")
mean = [ 546.7953,  649.5146,  967.6898, 1159.2480, 2442.1875, 2433.1270, 1822.9164]
std =  [ 447.3463,  519.1241,  668.2700,  904.4800, 1166.9471, 1226.1171, 1154.9677]
# 创建数据集实例
train_dataset = SsliDataset(train_paths, mean, std, ratio=0.3)
val_dataset = SsliDataset(val_paths, mean, std, ratio=0.4)
test_dataset = SsliDataset(test_paths, mean, std)

ratio = 0.3
batch_size = 350
num_workers = 8

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=False,
    num_workers=num_workers)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=False,
    num_workers=num_workers)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=False,
    num_workers=num_workers)

model = SSLI(7,256,4,8,7,0.2)
lit_model = LitModel(model, mean, std)


logger = TensorBoardLogger("tb_logs", name="my_model")

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='checkpoints/',
    filename='model-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    mode='min',
)

# 配置 EarlyStopping 回调
early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=3,
    mode='min'
)
# 创建 Trainer 并配置 logger
trainer = pl.Trainer(
    max_epochs=5,
    min_epochs=1,
    accelerator="gpu",
    devices=1,
    logger=logger,
    callbacks=[checkpoint_callback, early_stop_callback],
    precision=32,  # or 16 for mixed precision
    profiler="simple"  # Use the simple profiler
)

# 训练模型
trainer.fit(lit_model, train_loader, val_loader)

# 测试模型
trainer.test(dataloaders=test_loader)