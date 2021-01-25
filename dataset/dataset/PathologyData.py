from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage.filters import gaussian_filter, median_filter
from scipy.ndimage.interpolation import map_coordinates
from PIL import Image, ImageEnhance

'''
TAFE 阶段的训练的数据
'''
# 弹性变化
def elastic_transform(shape, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    # This function get indices only.
    if random_state is None:
        random_state = np.random.RandomState(None)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
    return indices


# Image Net mean and std
norm_mean=[0.485, 0.456, 0.406]
norm_std=[0.229, 0.224, 0.225]


class PathologyDataSet(Dataset):
    # 图像变形
    def transformImg(self,img):
        _, _, mod = img.shape
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

    def __init__(self,imgs,labels,bounds,idx_list=range(16),crop_size = (256,256)):
        super(PathologyDataSet,self).__init__()
        self.imgs = imgs[idx_list]
        self.labels = labels[idx_list]
        self.bounds = bounds[idx_list]
        # 需要对图片进行augment
        self.naug = 6 # origin*1 + rotate*3 + flip*2
        self.nimg = self.imgs.shape[0] # 数量
        self.crop_size = crop_size
        # 将原来数据里面的归一化
        self.labels[self.labels != 0] = 1
        self.bounds[self.bounds != 0] = 1

    def __getitem__(self,idx):
        iaug = int(np.mod(idx,self.naug)) # index of augment
        index = int(np.floor(idx/self.naug))# index 代表原始图片的位置
        np.random.seed()
        img = self.imgs[index].copy()
        h,w,mod = img.shape
        # 图像变形
        img = self.transformImg(img)     
        seg_label = self.labels[index].copy()
        boundary_label = self.bounds[index].copy()
        # 将语义分割的图像转换成0-1格式的
        seg_label[seg_label != 0] = 1
        # crop 随机裁剪图片
        sh = np.random.randint(0, h-self.crop_size[0]-1)
        sw = np.random.randint(0, w-self.crop_size[1]-1)
        img = img[:,sh:(sh+self.crop_size[0]), sw:(sw+self.crop_size[1])]
        seg_label = seg_label[sh:(sh+self.crop_size[0]), sw:(sw+self.crop_size[1])]
        boundary_label = boundary_label[sh:(sh+self.crop_size[0]), sw:(sw+self.crop_size[1])]
        # Aug
        if iaug <=3 and iaug >0: # 3*rorate
            # np.rot90(img,k,axes)
            # k: rorate time; axes: 旋转的轴，不应该带通道那个维度
            img = np.rot90(img, iaug, axes=(len(img.shape)-2, len(img.shape)-1))
            seg_label = np.rot90(seg_label, iaug, axes=(len(seg_label.shape)-2, len(seg_label.shape)-1))
            boundary_label = np.rot90(boundary_label, iaug, axes=(len(boundary_label.shape)-2, len(boundary_label.shape)-1))
        elif iaug >= 4:
            img = np.flip(img,len(img.shape)-(iaug-3))
            seg_label = np.flip(seg_label, len(seg_label.shape)-(iaug-3))
            boundary_label = np.flip(boundary_label, len(boundary_label.shape)-(iaug-3))

        # 最后来一个随机的柔性变形,对图像、语义分割、实例分割
        if np.random.rand()>=0.5:
            rnd_et = np.random.rand(2)
            indices = elastic_transform(seg_label.shape, int(rnd_et[0]*20), 5*(rnd_et[1]+1.0))
            for imod in range(mod):
                img[imod] = map_coordinates(img[imod].squeeze(), indices, order=1, mode='reflect').reshape(img[imod].shape)
            seg_label = map_coordinates(seg_label.squeeze(), indices, order=1, mode='reflect').reshape(seg_label.shape)
            boundary_label = map_coordinates(boundary_label.squeeze(), indices, order=1, mode='reflect').reshape(boundary_label.shape)
        img = img.copy()
        seg_label = seg_label.copy()
        boundary_label = boundary_label.copy()
        return img, seg_label, boundary_label
        # return self.imgs[idx], self.label[idx], self.bound[idx]

    def __len__(self):
        # number of image, number of augment
        return  self.nimg * self.naug 

    def seed(self):
        return np.random.rand()*0.1+0.9




