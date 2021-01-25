import sys
packages = [
    '../dataset/',
    '../utils/',
    '../optimizer/',
    '../loss/',
    '../model/'
]
sys.path.extend(packages)
from sklearn.model_selection import KFold
from tqdm import tqdm
from brpnet.model import UNet
from loss import dice_loss, smooth_truncated_loss, compute_loss_list
from adamw_r.cyclic_scheduler import CyclicLRWithRestarts, ReduceMaxLROnRestart
from adamw_r.adamw import AdamW
from ImageProcess.ImgShow import showLineImg
from dataset.PathologyData import PathologyDataSet
import cv2
import torch.nn.functional as F
import time
import torch
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn as nn
from postProcess.post_proc import post_proc
# 首先先对图片进行染色归一化处理
from StainProcess.StainNormalization import normalizeStaining
from ImageProcess.ImageProcess import cv2Bgr2Rgb
from DataUtil import kfold_list
from util import getNowTime,transformImg,extract_patches,reconstruct_from_patches_weightedall,test_extract_patches

# images = np.load('../dataset/trainDataSet.npy')
# labels = np.load('../dataset/labelDataSet.npy')
# bounds = np.load('../dataset/boundDataSet.npy')

test_same_images = np.load('../dataset/kumarDataset/train/testSameDataSet.npy')
test_same_labels = np.load('../dataset/kumarDataset/labels/labelSameDataSet.npy')
test_same_bounds = np.load('../dataset/kumarDataset/bounds/boundSameDataSet.npy')

CFG = {
    "batch_size": 1,
    "num_epoch":10,
    # 不同损失所占的额权重比
    "dice_weight":0.5
}

batch_size = 1
print('测试和标签的数据集大小: ',test_same_images.shape,test_same_labels.shape,test_same_bounds.shape)
# 对test_same的数据进行处理
def infer(model,device,image):
    aug_list = ['ori', 'rot90', 'rot180', 'rot270', 'flip_h', 'flip_w']
    image_tensor = torch.FloatTensor(image)
    image_tensor = image_tensor.unsqueeze(dim=0)
    image_tensor = image_tensor.permute(0,3,1,2)
    # (0,3,1000,1000)
    # print('image_tensor: ',image_tensor.shape)
    img_num,c,h,w = image_tensor.shape
    patches_d, ibs, shs, sws = extract_patches(image_tensor,256,128)
    sigmoid = nn.Sigmoid()
    # 定义最后的结果
    finalsout = np.zeros([1, h, w], dtype=np.float32)
    finalcout = np.zeros([1, h, w], dtype=np.float32)
    # tta操作
    for type_aug in aug_list:
        if type_aug == 'ori':
            tta_d = patches_d.clone()
        if type_aug == 'rot90':
            tta_d = patches_d.rot90(1, dims=(2,3))
        if type_aug == 'rot180':
            tta_d = patches_d.rot90(2, dims=(2,3))
        if type_aug == 'rot270':
            tta_d = patches_d.rot90(3, dims=(2,3))
        if type_aug == 'flip_h':
            tta_d = patches_d.flip(2)
        if type_aug == 'flip_w':
            tta_d = patches_d.flip(3)
        # 
        spred = torch.zeros(tta_d.shape[0],1,tta_d.shape[2],tta_d.shape[3])
        cpred = torch.zeros(tta_d.shape[0],1,tta_d.shape[2],tta_d.shape[3])

        for start_batch in list(range(0,tta_d.shape[0],batch_size)):
            end_batch = np.min([start_batch + batch_size, tta_d.shape[0]])
            input_d = tta_d[start_batch:end_batch]
            if len(input_d.shape) == 3:
                input_d = input_d.unsqueeze(dim=0)
            input_d = input_d.to(device)
            outs = model(input_d)
            sout = outs[0]
            cout = outs[5]
            # sigmod的原因在于将一些负数压制到0，然后相减的时候就可以直接操作了
            spred[start_batch:end_batch] = sigmoid(sout).data.cpu()
            cpred[start_batch:end_batch] = sigmoid(cout).data.cpu()
        # Inverse TTA
        if type_aug == 'rot90':
            spred = spred.rot90(3, dims=(len(spred.shape)-2,len(spred.shape)-1))
            cpred = cpred.rot90(3, dims=(len(cpred.shape)-2,len(cpred.shape)-1))
        elif type_aug == 'rot180':
            spred = spred.rot90(2, dims=(len(spred.shape)-2,len(spred.shape)-1))
            cpred = cpred.rot90(2, dims=(len(cpred.shape)-2,len(cpred.shape)-1))
        elif type_aug == 'rot270':
            spred = spred.rot90(1, dims=(len(spred.shape)-2,len(spred.shape)-1))
            cpred = cpred.rot90(1, dims=(len(cpred.shape)-2,len(cpred.shape)-1))
        elif type_aug == 'flip_h':
            spred = spred.flip(len(spred.shape)-2)
            cpred = cpred.flip(len(cpred.shape)-2)
        elif type_aug == 'flip_w':
            spred = spred.flip(len(spred.shape)-1)
            cpred = cpred.flip(len(cpred.shape)-1)
        #
        spred_map = reconstruct_from_patches_weightedall(spred, ibs, shs, sws, 256, 128, img_num, 1, w, h, 1).squeeze().numpy()
        cpred_map = reconstruct_from_patches_weightedall(cpred, ibs, shs, sws, 256, 128, img_num, 1, w, h, 1).squeeze().numpy()
        #
        finalsout += spred_map
        finalcout += cpred_map

    finalsout /= len(aug_list)
    finalcout /= len(aug_list)
    # 神经网络输出的结果
    return finalsout,finalcout




# 执行主程序的地方
if __name__ == "__main__":
    # 定义一些需要执行的变量
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(3, 1, 1).to(device)
    # train_model/model_of_770.pth
    # train_model/model_of_360.pth
    model.load_state_dict(torch.load('../train_model/model_of_360.pth'))
    for idx,img_input in enumerate(test_same_images):
        # 保存的数据包括原图、gt、sout、cout、pred
        gt = test_same_labels[idx]
        print('img_input shape: ',img_input.shape)
        sout,cout = infer(model,device,img_input)
        pred0 = post_proc(sout-cout,post_dilation_iter = 0)
        pred1 = post_proc(sout-cout,post_dilation_iter = 1)
        pred2 = post_proc(sout-cout,post_dilation_iter = 2)
        # 保存输出的值
        pred0 = pred0[0]
        pred1 = pred1[0]
        pred2 = pred2[0]
        # 保存的数据都是 (1000,1000)
        np.savez('test_same/val_{}_pred_model_of_360.npz'.format(idx), img=img_input, sout=sout[0], cout=cout[0], gt=gt, pred0=pred0, pred1=pred1, pred2=pred2)
        print('test_same/val_{}_pred_model_of_360.npz'.format(idx),'数据生成完成...')









