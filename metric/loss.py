import torch.nn as nn
import torch
class mask_loss(nn.Module): # 训练CNN用
    def __init__(self, mean, std, batch=True):
        super(mask_loss, self).__init__()
        self.batch = batch
        self.loss = nn.L1Loss(reduction='mean')
        self.mean = mean
        self.std = std
    
    def __call__(self, pred, target):
        y, mask = target[:,:,:7], target[:,:,7:]        
        mean = self.mean.cuda().to(pred.get_device())
        std = self.std.cuda().to(pred.get_device())
        pred = torch.sigmoid((pred*std+mean)/10000.)
        y = (y*std+mean)/10000.
        loss = self.loss(pred[mask==1.0], y[mask==1.0])
        change = torch.diff(pred, dim=1)
        smooth_loss = torch.abs(torch.diff(change,dim=1)).mean()
        return loss+smooth_loss*0.2