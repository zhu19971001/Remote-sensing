import numpy as np
from utils.oif_choices import oif_choose


class add_channels(object):
    def __init__(self, image):
        self.img = image
        self.img_oif = oif_choose(image, bands=6, choices=3)

    def img_add_channel(self, ndvi=False, exg=False, vdvi=False, rvi=False, ndgi=False, ngbdi=False, ndcsi=False):
        lis = []
        if ndvi:
            ndvi = self.NDVI()
            lis.append(ndvi)
        if exg:
            exg = self.EXG()
            lis.append(exg)
        if vdvi:
            vdvi = self.VDVI()
            lis.append(vdvi)
        if rvi:
            rvi = self.RVI()
            lis.append(rvi)
        if ndgi:
            ndgi = self.NDGI()
            lis.append(ndgi)
        if ngbdi:
            ngbdi = self.NGBDI()
            lis.append(ngbdi)
        if ndcsi:
            ndcsi = self.NDCSI()
            lis.append(ndcsi)

        spectral_channels = np.stack(lis)      # 各指数通道堆叠，维度增加
        img_channel_added = np.concatenate([self.img_oif, spectral_channels])     # 原图像和堆叠后的指数通道拼接
        return img_channel_added

    def NDVI(self):              # 归一化植被差异指数
        ndvi = (self.img[5] - self.img[2]) / (self.img[5] + self.img[2])
        return ndvi

    def EXG(self):               # 过绿指数：去除裸土、遮蔽物等非植被
        exg = self.img[1] * 2 - self.img[0] - self.img[1]
        return exg

    def VDVI(self):                                # 可见光差异植被指数
        vdvi = (self.img[1] * 2 - self.img[0] - self.img[1]) / (self.img[1] * 2 + self.img[0] + self.img[1])
        return vdvi

    def RVI(self):                               # 比值植被指数：区分果树与杂草
        rvi = self.img[5] / self.img[2]
        return rvi

    def NDGI(self):                                      # 归一化差异绿度指数：区分果树与杂草
        ndgi = (self.img[1] - self.img[2]) / (self.img[1] + self.img[2])
        return ndgi

    def NGBDI(self):                   # 归一化绿蓝差异指数
        g = (self.img[1] - self.img[1].min()) / (self.img[1].max() - self.img[1].min())
        b = (self.img[0] - self.img[0].min()) / (self.img[0].max() - self.img[0].min())
        ngbdi = (g - b) / (g + b)
        return ngbdi

    def NDCSI(self):                   # 归一化差异冠层阴影指数
        ndvi = (self.img[5] - self.img[2]) / (self.img[5] + self.img[2])
        ndcsi = ndvi * ((self.img[4] - self.img[4].min()) / (self.img[4].max() - self.img[4].min()))
        return ndcsi
