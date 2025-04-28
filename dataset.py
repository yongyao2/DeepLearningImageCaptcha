# -*- coding: UTF-8 -*-
import os

import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import encoding as ohe
import setting


class mydataset(Dataset):
    
    def __init__(self, folder, transform=None):
        self.train_image_file_paths = [os.path.join(folder, image_file) for image_file in os.listdir(folder)]
        self.transform = transform
    
    def __len__(self):
        return len(self.train_image_file_paths)
    
    def __getitem__(self, idx):
        image_root = self.train_image_file_paths[idx]
        image_name = image_root.split(os.path.sep)[-1]
        image = Image.open(image_root)
        if self.transform is not None:
            image = self.transform(image)
        label = ohe.encode(image_name.split('_')[0])
        return image, label

# 自定义颜色过滤函数
def filter_colors(image):
    img_array = np.array(image)
    mask = (
            (img_array[:, :, 0] != 255) |
            (img_array[:, :, 1] != 255) |
            (img_array[:, :, 2] != 255)
    )
    mask &= ~(
            (img_array[:, :, 0] == 179) &
            (img_array[:, :, 1] == 179) &
            (img_array[:, :, 2] == 255)
    )
    filtered_img = np.zeros_like(img_array)
    filtered_img[mask] = img_array[mask]
    return Image.fromarray(filtered_img)

# transform = transforms.Compose([
#     # transforms.ColorJitter(),
#     transforms.Lambda(lambda img: filter_colors(img)),
#     transforms.Grayscale(),# # 将彩色图像转为灰度图（单通道）
#     transforms.ToTensor(), # # 将 PIL 图像或 NumPy 数组转为 PyTorch 张量，并归一化到 [0, 1]
#     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# 定义预处理流水线
transform = transforms.Compose([
    #transforms.Lambda(lambda img: filter_colors(img)),   # 1. 自定义颜色过滤（去除背景和干扰线）
    transforms.Grayscale(), # 2. 转为灰度图（单通道）
    #transforms.ColorJitter(brightness=0.5, contrast=0.5),# 3. 调整对比度
    transforms.ToTensor(), # 4. 转为PyTorch张量
])


def get_train_data_loader():
    dataset = mydataset(setting.TRAIN_DATASET_PATH, transform=transform)
    return DataLoader(dataset, batch_size=64, shuffle=True)


def get_eval_data_loader():
    dataset = mydataset(setting.EVAL_DATASET_PATH, transform=transform)
    return DataLoader(dataset, batch_size=1, shuffle=True)


def get_predict_data_loader():
    dataset = mydataset(setting.PREDICT_DATASET_PATH, transform=transform)
    return DataLoader(dataset, batch_size=1, shuffle=True)
