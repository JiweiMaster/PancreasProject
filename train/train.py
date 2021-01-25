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
from util import getNowTime
from tqdm import tqdm
from brpnet.model import UNet
from loss import dice_loss, smooth_truncated_loss, compute_loss_list
from adamw_r.cyclic_scheduler import CyclicLRWithRestarts, ReduceMaxLROnRestart
from adamw_r.adamw import AdamW
from ImageProcess.ImgShow import showLineImg
from dataset.PathologyData import PathologyDataSet
from dataset.KumarPatchDataSet import KumarPatchDataSet
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
from metrics import get_fast_aji,remap_label,get_fast_aji_plus

# 导入自己定义的一些需要的包
print('获取到的当前是时间： ', getNowTime())

# 获取数据 训练的数据
# 原来的数据里面的语义分割的结果的值都是0,1二值的数值
# imgs = np.load('../dataset/trainDataSet.npy')
# labels = np.load('../dataset/labelDataSet.npy')
# bounds = np.load('../dataset/boundDataSet.npy')

# 训练的数据
imgs_train = np.load('../dataset/kumarDataset/train/trainDataSet.npy')
labels_train = np.load('../dataset/kumarDataset/labels/labelDataSet.npy')
bounds_train = np.load('../dataset/kumarDataset/bounds/boundDataSet.npy')

# 验证的数据
imgs_valid = np.load('../dataset/kumarDataset/train/testSameDataSet.npy')
labels_valid = np.load('../dataset/kumarDataset/labels/labelSameDataSet.npy')
bounds_valid = np.load('../dataset/kumarDataset/bounds/boundSameDataSet.npy')

# 一些超参数的设置
CFG = {
    "batch_size": 4,
    "num_epoch":1000,
    # 不同损失所占的额权重比
    "dice_weight":0.5
}

# 获取训练集的数据
def getTrainDataSet(imgs,labels,bounds,batch_size):
    pathologyData = PathologyDataSet(imgs, labels, bounds)
    pathologyDataloader = DataLoader(pathologyData, shuffle=True, batch_size=batch_size)
    return pathologyData,pathologyDataloader

# 获取验证集的数据
def getValDataSet(imgs,labels,bounds,batch_size):
    idxs = [0,1]
    imgs_val = imgs[idxs]
    imgs_val = np.transpose(imgs_val,(0,3,1,2)) # 训练的图像的数据
    # print('imgs_val: ',imgs_val.shape)
    labels_val = labels[idxs]
    labels_val = np.expand_dims(labels_val,4)
    labels_val = labels_val.transpose(0,3,1,2)
    # print('labels_val: ',labels_val.shape)
    bounds_val = bounds[idxs]
    bounds_val = np.expand_dims(bounds_val,4)
    bounds_val = bounds_val.transpose(0,3,1,2) # 训练的边界的数据
    # print('bounds_val: ',bounds_val.shape)
    # 生成数据集
    val_dataset = KumarPatchDataSet(imgs_val,labels_val,bounds_val)
    val_dataloader = DataLoader(val_dataset,shuffle=False,batch_size=batch_size)
    return val_dataset,val_dataloader

# 进行valid上面的验证    
def valid(model,device,dataloader):
    sigmod = nn.Sigmoid()
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    ajiMeanList = []
    ajiList= []
    for idx,(images,labels,bounds) in pbar:
        images = images.float().to(device)
        sout, sout_0, sout_1, sout_2, sout_3, cout, cout_0, cout_1, cout_2, cout_3 = model(images)
        # 计算损失函数
        # 计算aji
        # print('输出图片的大小: ',sout.shape)
        for index in range(sout.shape[0]):
            pred_sout = sigmod(sout[index][0]).cpu().data.numpy()
            pred_cout = sigmod(cout[index][0]).cpu().data.numpy()
            gt_label0 = labels[index][0]
            # 格式化label
            gt_label0[gt_label0 != 0] = 1
            # 后处理图像并且计算值
            # 这个是TAFE需要进行的操作
            postProcImg = post_proc(pred_sout - pred_cout, post_dilation_iter=2)
            # 解决一些训练的时候产生的错误
            if postProcImg is not None:
                aji = get_fast_aji(remap_label(gt_label0), remap_label(postProcImg))
                print(aji)
                ajiList.append(aji)
        # 全部计算完成之后计算一次数据
        ajiArray = np.array(ajiList)
        ajiMean = np.mean(ajiArray)
        pbar.set_description(f'valid: ----ajiMean---- {ajiMean}')
        ajiMeanList.append(ajiMean)
    return ajiMeanList
    
#验证集的测试代码
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(3, 1, 1).to(device)
# train_model/segModel_epoch_15.pth
# train_model/model_of_770.pth
model.load_state_dict(torch.load('../train_model/model_of_770.pth'))
val_dataset,val_dataloader = getValDataSet(imgs_valid,labels_valid,bounds_valid,1)
aji = valid(model,device,val_dataloader)
ajimean = np.mean(aji)
print(ajimean)

# 定义一个简单的初始化函数，训练模型的参数,这是一次全量数据集的训练
def train(model, device, dataloader, lossfn, optimizer, scheduler,dice_weight=0.5):
    model.train()
    scheduler.step()
    runningloss = 0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    sample_num = 0
    lossesList = []
    for index, (image, label, boundary) in pbar:
        b, _, h, w = image.shape
        # 输入数据
        image = image.float().to(device)
        label = label.float().to(device).view(b, 1, h, w)
        boundary = boundary.float().to(device).view(b, 1, h, w)
        # 
        label_s1 = F.interpolate(label, scale_factor=0.5, mode='bilinear')
        label_s2 = F.interpolate(label, scale_factor=0.25, mode='bilinear')
        label_s3 = F.interpolate(label, scale_factor=0.125, mode='bilinear')
        boundary_s1 = F.interpolate(boundary, scale_factor=0.5, mode='bilinear')
        boundary_s2 = F.interpolate(boundary, scale_factor=0.25, mode='bilinear')
        boundary_s3 = F.interpolate( boundary, scale_factor=0.125, mode='bilinear')
        # 计算损失
        sout, sout_0, sout_1, sout_2, sout_3, cout, cout_0, cout_1, cout_2, cout_3 = model(image)
        seg_stl_losses = lossfn(smooth_truncated_loss, [sout, sout_0, sout_1, sout_2, sout_3], [label, label, label_s1, label_s2, label_s3])
        seg_dsc_losses = lossfn(dice_loss, [sout], [label])
        bnd_stl_losses = lossfn(smooth_truncated_loss, [cout, cout_0, cout_1, cout_2, cout_3], [boundary, boundary, boundary_s1, boundary_s2, boundary_s3])
        bnd_dsc_losses = lossfn(dice_loss, [cout], [boundary])
        # 定义loss的值
        loss = 0.0
        for iloss in seg_stl_losses:
            loss += iloss
        for iloss in seg_dsc_losses:
            loss += iloss*dice_weight
        for iloss in bnd_stl_losses:
            loss += iloss
        for iloss in bnd_dsc_losses:
            loss += iloss*dice_weight
        # 更新参数
        loss.backward() 
        optimizer.step()  # 这个才是更新模型
        scheduler.batch_step() # 更新学习率
        # 计算损失的和
        runningloss = runningloss*0.99 + loss.item()*0.01
        if((index+1) % 10):
            pbar.set_description(f'full train index: [{index+1} -- {len(dataloader)}] -- loss: {runningloss}')
            lossesList.append(runningloss)
    return lossesList


# 使用验证集来验证模型的参数
'''
if __name__ == "__main__":
    if_use_mult_device = False
    ############################################ 直接用训练集进行训练，在test_same上面进行验证模型的效果来选择合适的模型 ###########################################
    # 训练数据集
    pathologyData,pathologyDataloader = getTrainDataSet(imgs_train,labels_train,bounds_train,CFG['batch_size'])
    # 验证数据集
    val_dataset,val_dataloader = getValDataSet(imgs_valid,labels_valid,bounds_valid,1)
    # 定义模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        if_use_mult_device = True
        model = nn.DataParallel(UNet(3, 1, 1).to(device),device_ids=[0,1])
    else:
        if_use_mult_device = False
        model = UNet(3, 1, 1).to(device)
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    print('scheduler batchsize: ',CFG['batch_size'],'len(pathologyData): ',len(pathologyData),'epoch size: ',CFG['num_epoch'])
    scheduler = CyclicLRWithRestarts(optimizer, CFG['batch_size'],len(pathologyData), restart_period=50, t_mult=2.0, policy="cosine", eta_on_restart_cb=ReduceMaxLROnRestart(0.5))
    model.train()

    # ##########################################################################开始训练###############################################################
    ajiList = []
    lossListArray = []
    
    for epoch in range(CFG['num_epoch']):
        print('->', str(epoch)+'th training epoch, '+'cur_lr '+str(optimizer.param_groups[0]['lr']))
        # 每一次训练的损失下来
        lossList = train(model, device, pathologyDataloader, compute_loss_list, optimizer, scheduler)
        lossListArray.append(lossList)
        # 保存损失
        np.save(str(epoch)+'_loss.npy',lossList)
        # 保存模型的参数
        if (epoch+1) % 10 == 0:
            if if_use_mult_device:
                torch.save(model.module.state_dict(), '../train_model/segModel_epoch_'+str(epoch)+'.pth')
            else:
                torch.save(model.state_dict(), '../train_model/segModel_epoch_'+str(epoch)+'.pth')
        ################################################################### 进行模型的验证 ##################################################################
            with torch.no_grad():
                aji = valid(model,device,val_dataloader)
                ajiList.append(aji)
        # 保存losses的值
        np.save(f'../train_model/lossListArray_{epoch}.npy',lossListArray)
        np.save(f'../train_model/ajiList_{epoch}.npy',ajiList)
'''

'''
# 
dice_weight = CFG['dice_weight']
batch_size = CFG['batch_size']
# 是否使用多个GPU
if_use_mult_device = False

if torch.cuda.device_count() > 1:
    if_use_mult_device = True
else:
    if_use_mult_device = False

imgs_train = np.load('../dataset/kumarDataset/train/trainDataSet.npy')
labels_train = np.load('../dataset/kumarDataset/labels/labelDataSet.npy')
bounds_train = np.load('../dataset/kumarDataset/bounds/boundDataSet.npy')
pathologyData,pathologyDataloader = getTrainDataSet(imgs_train,labels_train,bounds_train,CFG['batch_size'])
# 验证数据集
val_dataset,val_dataloader = getValDataSet(imgs_valid,labels_valid,bounds_valid,1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(3,1,1).to(device)

if if_use_mult_device:
    model = nn.DataParallel(model,device_ids=[0,1])

optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
scheduler = CyclicLRWithRestarts(optimizer, batch_size, len(pathologyData),restart_period=50, \
    t_mult=2.0, policy="cosine", eta_on_restart_cb = ReduceMaxLROnRestart(0.5))

slosses = []
closses = []
dslosses = []
dclosses = []

ajiList = []
lossList = []

for epoch in range(CFG['num_epoch']):
    scheduler.step()
    slosses_insideepoch = []
    closses_insideepoch = []
    dslosses_insideepoch = []
    dclosses_insideepoch = []
    model.train()
    print('->', str(epoch)+'th training epoch, '+'cur_lr '+str(optimizer.param_groups[0]['lr']))
    losses = []
    for i_data,(image, label, boundary) in enumerate(pathologyDataloader):
        st = time.time()
        optimizer.zero_grad()
        b, _, h, w = image.shape
        image = image.float().cuda()
        label = label.float().cuda().view(b, 1, h, w)
        boundary = boundary.float().cuda().view(b, 1, h, w)

        label_s1 = F.interpolate(label, scale_factor=0.5, mode='bilinear')
        label_s2 = F.interpolate(label, scale_factor=0.25, mode='bilinear')
        label_s3 = F.interpolate(label, scale_factor=0.125, mode='bilinear')
        boundary_s1 = F.interpolate(boundary, scale_factor=0.5, mode='bilinear')
        boundary_s2 = F.interpolate(boundary, scale_factor=0.25, mode='bilinear')
        boundary_s3 = F.interpolate(boundary, scale_factor=0.125, mode='bilinear')

        sout, sout_0, sout_1, sout_2, sout_3, cout, cout_0, cout_1, cout_2, cout_3 = model(image)
        # 计算损失
        seg_stl_losses = compute_loss_list(smooth_truncated_loss, [sout, sout_0, sout_1, sout_2, sout_3], [label, label, label_s1, label_s2, label_s3])
        seg_dsc_losses = compute_loss_list(dice_loss, [sout], [label])
        bnd_stl_losses = compute_loss_list(smooth_truncated_loss, [cout, cout_0, cout_1, cout_2, cout_3], [boundary, boundary, boundary_s1, boundary_s2, boundary_s3])
        bnd_dsc_losses = compute_loss_list(dice_loss, [cout], [boundary])
        loss = 0.0
        for iloss in seg_stl_losses:
            loss += iloss
        for iloss in seg_dsc_losses:
            loss += iloss*dice_weight
        for iloss in bnd_stl_losses:
            loss += iloss
        for iloss in bnd_dsc_losses:
            loss += iloss*dice_weight

        loss.backward()
        optimizer.step()
        scheduler.batch_step()

        et = time.time()
        print('epoch {0: d}, batch {1:d}, loss {2: .6f}, time {2: .4f}'.format(epoch, i_data, loss, et-st))
        losses.append(loss.item())

    # 每次epoch保存一次
    lossList.append(np.mean(np.array(losses)))
    if (epoch+1) % 10 == 0:
        if if_use_mult_device:
            print('epoch: ',epoch+1,'保存训练的参数到本地...')
            torch.save(model.module.state_dict(),'../train_model/model_of_'+str(epoch+1)+'.pth')
        else:
            print('epoch: ',epoch+1,'保存训练的参数到本地...')
            torch.save(model.state_dict(),'../train_model/model_of_'+str(epoch+1)+'.pth')

        # 验证集上面进行验证
        with torch.no_grad():
            aji = valid(model,device,val_dataloader)
            print('epoch: ',epoch,'--aji: ',aji)
            ajiList.append(aji)

        #保存loss数据
        np.save(f'{epoch}_loss.npy',np.array(lossList))
        np.save(f'{epoch}_aji.npy',np.array(ajiList))

'''
        