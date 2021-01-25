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
from metrics import get_fast_aji,remap_label
from scipy import ndimage
from skimage.morphology import label
import scipy.io as scio


'''
执行这一步的时候首先先用模型生成数据，然后进行aji参数的计算
'''

# 计算aji的值的大小
def calculateAJI(sout,cout,gt_lbl):
    lbl = post_proc(sout-cout, post_dilation_iter=1)
    aji = get_fast_aji(remap_label(gt_lbl), remap_label(lbl))
    return aji


# 主函数的运行方式
def main():
    # 先获取数据集的预测值的大小
    test_same_train = './test_same'
    test_same_lbl = '../dataset/kumarDataset/labels/labelSameDataSet.npy'
    gt_Labels = np.load(test_same_lbl)
    ajiList = []
    for index, predImg in enumerate(sorted(os.listdir(test_same_train))):
        trainImage = test_same_train+'/'+predImg
        outputImg = np.load(trainImage)
        sout = outputImg['sout']
        cout = outputImg['cout']
        gt_label = gt_Labels[index]
        # 计算aji的数值的大小
        aji_num = calculateAJI(sout,cout,gt_label)
        ajiList.append(aji_num)
    print(ajiList)



def calculate():
    mylist_dilation2 = [0.5595103276451643,0.47205656333881724,0.5257287705956908,0.5786233446025155,0.640603078546504,0.6210917037716082,0.6000145071205784,0.6519124332276229]
    mylist_dilation1 = []
    mylist_dilation0 = [0.43010154827514574, 0.36713685320991646, 0.37857465076207447, 0.39005596689550204, 0.5626010047779652, 0.5090649479944458, 0.4931520637753904, 0.5669585657766227]
    myArray = np.array(mylist_dilation2)
    result = np.mean(myArray)
    return result

if __name__ == "__main__":
    main()
    # print(calculate())
    






