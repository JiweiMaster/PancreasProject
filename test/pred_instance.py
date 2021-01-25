import sys
packages = [
    '../dataset/',
    '../utils/',
    '../optimizer/',
    '../loss/',
    '../model/',
    '../patchNet'
]
sys.path.extend(packages)
from sklearn.model_selection import KFold
from tqdm import tqdm
# from brpnet.model import UNet
from patch_dense_net import UNet
from loss import dice_loss, smooth_truncated_loss, compute_loss_list
from adamw_r.cyclic_scheduler import CyclicLRWithRestarts, ReduceMaxLROnRestart
from adamw_r.adamw import AdamW
from ImageProcess.ImgShow import showLineImg
from dataset.PathologyData import PathologyDataSet
from dataset.InstanceDataset import InstanceDataset
import cv2
import torch.nn.functional as F
import time
import torch
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn as nn
from postProcess.post_proc import post_proc,getInstancePosi,getInstanceArray
# 首先先对图片进行染色归一化处理
from StainProcess.StainNormalization import normalizeStaining
from ImageProcess.ImageProcess import cv2Bgr2Rgb
from DataUtil import kfold_list
from util import getNowTime,transformImg,extract_patches,reconstruct_from_patches_weightedall,test_extract_patches
from scipy import ndimage


def getData():
    trains = np.load('../dataset/kumarDataset/train/testSameDataSet.npy')
    gts = np.load('../dataset/kumarDataset/labels/labelSameDataSet.npy')
    val_pred = np.load('test_same/val_0_pred.npz')
    train0 = trains[0]
    sout = val_pred['sout']
    cout = val_pred['cout']
    pred = val_pred['pred']
    gt0 = gts[0]
    return train0,sout,cout,pred,gt0

img,sout,cout,pred,gt = getData()
insDataset = InstanceDataset(img,sout,cout,pred,gt)


batch_size = 1
dataload = DataLoader(insDataset, shuffle = False, batch_size = batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model48 = UNet(3,2,1).to(device)
model176 = UNet(3,2,1).to(device)
# patchNet/size48_model_epoch_9.pth
model48.load_state_dict(torch.load('../patchNet/size48_model_epoch_9.pth'),device)
model176.load_state_dict(torch.load('../patchNet/size176_model_epoch_9.pth'),device)


sigmod = nn.Sigmoid()

instances = []
instances.append(np.zeros([1, 1176, 1176])+0.5)


for idx,(patch_imgs,patch_sins,patch_cins,patch_gts,sxs,sys,exs,eys) in enumerate(dataload):
    #这边需要注意数据是如何
    sxs,sys,exs,eys = sxs.numpy(),sys.numpy(),exs.numpy(),eys.numpy()
    patch_ins = torch.cat((patch_sins.unsqueeze(dim=1),patch_cins.unsqueeze(dim=1)),dim=1)
    patch_imgs = patch_imgs.to(device)
    patch_ins = patch_ins.to(device)
    if patch_imgs.shape[2] == 48:
        preds = model48(patch_imgs,patch_ins)
    else:
        preds = model176(patch_imgs,patch_ins)
    # pred_img0 = sigmod(pred[0][0]).cpu().data.numpy()
    # pred_img1 = sigmod(pred[1][0]).cpu().data.numpy()
    # showLineImg([pred_img0,pred_img1])
    probmap_instances = np.zeros([1, 1176, 1176])
    for i in range(batch_size):
        img_size = exs[i] - sxs[i]
        pred_img = sigmod(preds[i][0]).cpu().data.numpy()
        pred_img_resize = cv2.resize(pred_img,(img_size,img_size))
        probmap_instances[:,sxs[i]:exs[i],sys[i]:eys[i]] = pred_img_resize
        instances.append(probmap_instances.astype(np.float32)) # (1176,1176)

instances = np.concatenate(instances, axis=0)
maxprob_instances = np.max(instances, axis=0)
idx_instances = np.argmax(instances, axis=0)

np.save('instances_pred.npy',instances)
np.save('maxprob_instances.npy',maxprob_instances)
np.save('idx_instances.npy',idx_instances)
print(instances.shape,maxprob_instances.shape,idx_instances.shape)

