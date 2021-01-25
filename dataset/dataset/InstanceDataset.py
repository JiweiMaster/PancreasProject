from torch.utils.data import DataLoader,Dataset
import numpy as np
from scipy import ndimage
from PIL import Image, ImageEnhance
import cv2
import torch


# Image Net mean and std
norm_mean=[0.485, 0.456, 0.406]
norm_std=[0.229, 0.224, 0.225]

# 未使用opencv的方法获取instance的坐标
def getInstanceArray(gt_label):
    # 所有的instance的大小
    predset  = np.unique(gt_label[gt_label>0])
    instanceList = []
    for ic in predset:
        instance = gt_label.copy()
        instance[ instance != ic ] = 0
        instance[ instance == ic ] = 1
        # 获取不为0的位置
        icx, icy = np.nonzero(instance)
        maxx = icx.max()
        maxy = icy.max()
        minx = icx.min()
        miny = icy.min()
        mx = np.round((maxx+minx)/2)
        my = np.round((maxy+miny)/2)
        halfsz = (np.max([(maxx-minx)/2, (maxy-miny)/2, 8])+12).astype(np.int16)
        # 图像实际的起始位置
        sx = np.round(mx - halfsz).astype(np.int16)
        sy = np.round(my - halfsz).astype(np.int16)
        ex = np.round(mx + halfsz + 1).astype(np.int16)
        ey = np.round(my + halfsz + 1).astype(np.int16)
        # 添加instance的数据的，获取的是方形的数据
        instanceList.append([sx,sy,ex,ey])
    return instanceList


class InstanceDataset(Dataset):
    def __init__(self, img, sin, cin, pred_ins,gt=None, dilation_rate=2, dilation_s=0, dilation_c=0):
        super(InstanceDataset,self).__init__()
        # 获取所有的instance
        self.size48 = 48
        self.size176 = 176
        self.maxpad = 176 // 2
        self.img = np.pad(img,((self.maxpad,self.maxpad),(self.maxpad,self.maxpad),(0,0)),'reflect')
        self.sin = np.pad(sin,((self.maxpad,self.maxpad),(self.maxpad,self.maxpad)),'constant',constant_values=0)
        self.cin = np.pad(cin,((self.maxpad,self.maxpad),(self.maxpad,self.maxpad)),'constant',constant_values=0)
        self.pred_ins = np.pad(pred_ins,((self.maxpad,self.maxpad),(self.maxpad,self.maxpad)),'constant',constant_values=0)
        self.gt = gt
        if self.gt is not None:
            self.gt = np.pad(gt,((self.maxpad,self.maxpad),(self.maxpad,self.maxpad)),'constant',constant_values=0)
        # 获取所有的instance posi
        self.instances = getInstanceArray(self.pred_ins)
        # 获取所有instance对于的数值的数组
        self.cset = np.unique(pred_ins[pred_ins>0])
        # 一些后处理的参数
        self.dilation_rate = dilation_rate
        self.dilation_s = dilation_s
        self.dilation_c = dilation_c

    def __len__(self):
        return len(self.instances)

    def __getitem__(self,idx):
        ins_num = self.cset[idx]
        icmap = self.pred_ins.copy()
        icmap[icmap != ins_num] = 0
        icmap[icmap == ins_num] = 1
        sx,sy,ex,ey = self.instances[idx]
        # 
        if self.dilation_rate > 0:
            icmap = ndimage.morphology.binary_dilation(icmap, iterations=self.dilation_rate)
        # 
        if self.dilation_s > 0:
            dicmap_s = ndimage.morphology.binary_dilation(icmap, iterations=self.dilation_s)
        elif self.dilation_s == 0:
            dicmap_s = icmap.copy()
        else:
            dicmap_s = np.ones(icmap.shape)
        
        if self.dilation_c > 0:
            dicmap_c = ndimage.morphology.binary_dilation(icmap, iterations=self.dilation_c)
        elif self.dilation_c == 0:
            dicmap_c = icmap.copy()
        else:
            dicmap_c = np.ones(icmap.shape)
        # 获取真正需要进行处理的图片
        patch_img = self.img[sx:ex,sy:ey,:].astype(np.float32)
        # 需要获取的gt
        patch_gt = None
        if self.gt is not None:
            patch_gt = self.gt[sx:ex,sy:ey].astype(np.float32)
        patch_sin = self.sin[sx:ex,sy:ey].astype(np.float32) * dicmap_s[sx:ex, sy:ey].astype(np.float32)
        patch_cin = self.cin[sx:ex,sy:ey].astype(np.float32) * dicmap_c[sx:ex, sy:ey].astype(np.float32)
        # 将图片放大
        img_size = patch_sin.shape[0]
        if img_size < self.size48:
            patch_img = cv2.resize(patch_img,(self.size48,self.size48))
            patch_sin = cv2.resize(patch_sin,(self.size48,self.size48))
            patch_cin = cv2.resize(patch_cin,(self.size48,self.size48))
            if patch_gt is not None:
                patch_gt = cv2.resize(patch_gt,(self.size48,self.size48))
        if img_size > self.size48 and img_size < self.size176:
            patch_img = cv2.resize(patch_img,(self.size176,self.size176))
            patch_sin = cv2.resize(patch_sin,(self.size176,self.size176))
            patch_cin = cv2.resize(patch_cin,(self.size176,self.size176))
            if patch_gt is not None:
                patch_gt = cv2.resize(patch_gt,(self.size176,self.size176))
        # 
        patch_img = patch_img.astype('uint8')
        # 输出的结果都是可以直接显示的，没有对图像进行处理的操作
        return self.transformImg(patch_img),patch_sin,patch_cin,patch_gt,sx,sy,ex,ey

    # seed
    def seed(self): 
        return np.random.rand()*0.1+0.9
        
    # 图像变形
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