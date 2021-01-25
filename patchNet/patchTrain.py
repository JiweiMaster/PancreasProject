import sys
packages = [
    '../dataset/dataset/',
    '../utils/',
    '../optimizer/',
    '../loss/',
    '../model/'
]
sys.path.extend(packages)

from sklearn.model_selection import KFold
from util import getNowTime
from tqdm import tqdm
# from brpnet.model import UNet
from loss import dice_loss, smooth_truncated_loss, compute_loss_list,focal_loss,dice_loss_perimg
from adamw_r.cyclic_scheduler import CyclicLRWithRestarts, ReduceMaxLROnRestart
from adamw_r.adamw import AdamW
from ImageProcess.ImgShow import showLineImg
from PathologyData import PathologyDataSet
from KumarPatchDataSet import KumarPatchDataSet
import cv2
import torch.nn.functional as F
import time
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn as nn
from DataUtil import kfold_list
from postProcess.post_proc import post_proc
from metrics import get_fast_aji,remap_label
from dataset import PatchDataset
from patch_dense_net import UNet

# patchNet/size48/size48_train_npy.npy

size176_train_npy = np.load('size48/size48_train_npy.npy')
size176_sout_ins_npy = np.load('size48/size48_sout_ins_npy.npy')
size176_cout_ins_npy = np.load('size48/size48_cout_ins_npy.npy')
size176_pred_ins_npy = np.load('size48/size48_pred_ins_npy.npy') 
size176_label_ins_npy = np.load('size48/size48_label_ins_npy.npy')
patch_dataset = PatchDataset(size176_train_npy,size176_sout_ins_npy,size176_cout_ins_npy,size176_pred_ins_npy,size176_label_ins_npy)

# size176_train_npy = np.load('size176/size176_train_npy.npy')
# size176_sout_ins_npy = np.load('size176/size176_sout_ins_npy.npy')
# size176_cout_ins_npy = np.load('size176/size176_cout_ins_npy.npy')
# size176_pred_ins_npy = np.load('size176/size176_pred_ins_npy.npy') 
# size176_label_ins_npy = np.load('size176/size176_label_ins_npy.npy')
# patch_dataset = PatchDataset(size176_train_npy,size176_sout_ins_npy,size176_cout_ins_npy,size176_pred_ins_npy,size176_label_ins_npy)


batch_size = 16
niter_perepoch = len(patch_dataset)//batch_size
MAX_epoch = 10
num_steps = [100, 200]
num_step = num_steps[0]
weight_decay = 1e-5
lr = 3e-4
dice_weight = 0.5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNet(3,2,1).to(device)
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
nstep = np.ceil(MAX_epoch*niter_perepoch/num_step)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, nstep, eta_min=0, last_epoch=-1)
dataloader = DataLoader(patch_dataset,batch_size=batch_size,shuffle=True, num_workers=4)

for epoch in range(MAX_epoch):
    slosses_insideepoch = []
    for i_data,(image, label, s_in, c_in) in enumerate(dataloader):
        st = time.time()
        image = image.float().to(device)
        label = label.float().unsqueeze(dim=1).to(device)
        s_in = s_in.float().unsqueeze(dim=1).to(device)
        c_in = c_in.float().unsqueeze(dim=1).to(device)
        b, _, h, w = image.shape
        
        image_in = torch.cat((s_in, c_in),dim=1)
        print(image_in.shape)
        sout = model(image, image_in)
        loss = focal_loss(sout, label) + dice_weight*dice_loss_perimg(F.sigmoid(sout), label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        slosses_insideepoch.append(loss.item())
        et = time.time()
        print('epoch {0: d}, batch {1:d} ; sloss {3:.3f}; used {2:.6f} s'.format(epoch, i_data, et-st, loss))
    # 保存模型
    torch.save(model.state_dict(),f"model_epoch_{epoch}.pth")



