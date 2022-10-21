import scipy.io as sio
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import os
import cv2
from PIL import Image
import torchvision

class myDatasets(Dataset):
    def __init__(self,img_path, ann_path, down_sample=False,transform=None):
        self.pre_img_path = img_path
        self.pre_ann_path = ann_path
        # 图像的文件名是 IMG_15.jpg 则 标签是 GT_IMG_15.mat
        # 因此不需要listdir标签路径
        self.img_names = os.listdir(img_path)
        self.transform=transform
        self.down_sample = down_sample

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_name = self.img_names[index]
        mat_name = 'GT_' + img_name.replace('jpg','mat')

        img = Image.open(os.path.join(self.pre_img_path,img_name)).convert('RGB')
        img = np.array(img).astype(np.float32)
        
        # print(F"{h=},{w=}")
        if self.transform != None:
            img=self.transform(img)
        # img.permute(0,2,1) # totensor会自动进行维度的转换，所以这里是不必要的

        h,w = img.shape[1],img.shape[2]

        anno = sio.loadmat(self.pre_ann_path + mat_name)
        xy = anno['image_info'][0][0][0][0][0]  # N,2的坐标数组
        density_map = self.get_density((h,w), xy).astype(np.float32) # 密度图
        density_map = torch.from_numpy(density_map)

        return img,density_map


    def get_density(self,img_shape, points):
        h, w  = img_shape[0], img_shape[1]
        # 密度图 初始化全0
        labels = np.zeros(shape=(h,w))
        for loc in points:
            f_sz = 17  # 滤波器尺寸 预设为15 也是邻域的尺寸
            sigma = 4.0  # sigma参数
            H = self.fspecial(f_sz, f_sz , sigma)  # 高斯核矩阵
            x = min(max(0,abs(int(loc[0]))),int(w))  # 头部坐标
            y = min(max(0,abs(int(loc[1]))),int(h))
            if x > w or y > h:
                continue
            x1 = x - f_sz/2 ; y1 = y - f_sz/2
            x2 = x + f_sz/2 ; y2 = y + f_sz/2
            dfx1 = 0; dfy1 = 0; dfx2 = 0; dfy2 = 0

            change_H = False
            if x1 < 0:
                dfx1 = abs(x1);x1 = 0 ;change_H = True
            if y1 < 0:
                dfy1 = abs(y1); y1 = 0 ; change_H = True
            if x2 > w:
                dfx2 = x2-w ; x2 =w-1 ; change_H =True
            if y2 > h:
                dfy2 = y2 -h ; y2 = h-1 ; change_H =True
            x1h =  1 + dfx1 ; y1h =  1 + dfy1
            x2h = f_sz - dfx2 ; y2h = f_sz - dfy2
            if change_H :
                H = self.fspecial(int(y2h-y1h+1), int(x2h-x1h+1),sigma)
            labels[int(y1):int(y2), int(x1):int(x2)] = labels[int(y1):int(y2), int(x1):int(x2)] + H
        if self.down_sample:
            labels = cv2.resize(labels,(w//4,h//4))
        return labels

    def fspecial(self,ksize_x=5, ksize_y = 5, sigma=4):
        kx = cv2.getGaussianKernel(ksize_x, sigma)
        ky = cv2.getGaussianKernel(ksize_y, sigma)
        return np.multiply(kx,np.transpose(ky))


# for test
if __name__ == '__main__':
    path1 = r"C:\Users\ocean\Downloads\datasets\ShanghaiTech\part_A\train_data\images\\"
    path2 = r"C:\Users\ocean\Downloads\datasets\ShanghaiTech\part_A\train_data\ground-truth\\"
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    datasets = myDatasets(path1, path2,transform=transform)
    train_loader = DataLoader(datasets, batch_size=1) 
    x1,y1=None,None
    for x,y in train_loader:
        x1=x
        y1=y
        break
    print(x1.shape)
    print(y1.shape)