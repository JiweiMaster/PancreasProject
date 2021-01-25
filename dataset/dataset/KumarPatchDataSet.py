import os 
import sys
# 添加一些包的路径
packages = [
    '../dataset/',
    '../utils/',
    '../optimizer/',
    '../loss/',
    '../model/'
]
sys.path.extend(packages)
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage.filters import gaussian_filter, median_filter
from scipy.ndimage.interpolation import map_coordinates
from PIL import Image, ImageEnhance
from util import extract_patches
import torch

# Image Net mean and std
norm_mean=[0.485, 0.456, 0.406]
norm_std=[0.229, 0.224, 0.225]


# 代码还没有写完
class KumarPatchDataSet(Dataset):
    # 输入的数据都是4维度的tensor
    def __init__(self,imgs,labels,bounds):
        super(KumarPatchDataSet,self).__init__()
        self.imgs = torch.tensor(imgs).float()
        self.labels = torch.tensor(labels).float()
        self.bounds = torch.tensor(bounds).float()
        self.idx_array = list(np.arange(64).reshape(8,8)[1:7,1:7].reshape(36,).astype('int'))

        # 提取patches数据出来
        self.img_patches, self.img_ibs, self.img_shs, self.img_sws = extract_patches(self.imgs,256,128)
        self.label_patches, self.label_ibs, self.label_shs, self.label_sws = extract_patches(self.labels,256,128)
        self.bound_patches, self.bound_ibs, self.bound_shs, self.bound_sws = extract_patches(self.bounds,256,128)
    
    def __getitem__(self,index):
        # 每张大图产生36个patch
        a = index//36
        b = index%36
        # 映射到对应的空间里面去
        idx = a*64 + self.idx_array[b]
        idx = int(idx)
        img_patch = self.img_patches[idx]
        label_patch = self.label_patches[idx]
        bound_patch = self.bound_patches[idx]
        # 返回的就是训练的数据和真实的标签数据
        return img_patch,label_patch,bound_patch

    def __len__(self):
        return len(self.idx_array)*self.imgs.shape[0]-1

    def seed(self): 
        return np.random.rand()*0.1+0.9

    def transformImg(self,img):
        _, _, mod = img.shape
        if torch.is_tensor(img):
            img = img.numpy()
        # 一般的图像处理的过程
        img = Image.fromarray(img.astype(np.uint8))
        img = ImageEnhance.Color(img).enhance(self.seed())
        img = ImageEnhance.Brightness(img).enhance(self.seed())
        img = ImageEnhance.Contrast(img).enhance(self.seed())
        img = ImageEnhance.Sharpness(img).enhance(self.seed())
        img = np.asarray(img).astype(np.float32)
        img = img.transpose([2, 0, 1])
        for imod in list(range(mod)):
            img[imod] = (img[imod]/255.0 - norm_mean[imod])/norm_std[imod]
        img += np.random.normal(0, np.random.rand(), img.shape)*0.01
        return img