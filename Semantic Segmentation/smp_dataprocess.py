import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import torch
import glob
from utils.read_write import readtif
from utils.add_channels import add_channels
from utils.onehot import one_hot


class MyDataset(data.Dataset):
    def __init__(self, img_path, lab_path, lab_folder, mean, std,
                 ndvi, exg, vdvi, rvi, ndgi, ngbdi, ndcsi):

        self.img_path_list = glob.glob(img_path)
        self.lab_path_list = glob.glob(lab_path)
        self.lab_folder = lab_folder
        self.mean = mean
        self.std = std
        self.ndvi = ndvi
        self.exg = exg
        self.vdvi = vdvi
        self.rvi = rvi
        self.ndgi = ndgi
        self.ngbdi = ngbdi
        self.ndcsi = ndcsi

    def __getitem__(self, index):
        img = readtif(self.img_path_list[index])
        lab = readtif(self.lab_path_list[index])    # 灰度图h*w
        lab_ori = lab
                                                     # 添加通道
        # ---------------------------------------------------------------------------------------------------
        img = add_channels(img)
        img = img.img_add_channel(self.ndvi, self.exg, self.vdvi, self.rvi, self.ndgi, self.ngbdi, self.ndcsi)
        # ----------------------------------------------------------------------------------------------------
        img = np.uint8(img)      # uint8
        img = img.swapaxes(1, 0).swapaxes(1, 2)   # chw->hwc
        img = self.process(img)
        lab = one_hot(lab, self.lab_folder, classNum=2)
        lab = lab.swapaxes(1, 2).swapaxes(1, 0)
        lab = torch.tensor(lab).type(torch.float)   # 32为浮点型
        return img, lab, lab_ori

    def __len__(self):
        return len(self.img_path_list)

    def process(self, data):
        transforms_list = [
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ]
        data_transforms = transforms.Compose(transforms_list)
        return data_transforms(data)
