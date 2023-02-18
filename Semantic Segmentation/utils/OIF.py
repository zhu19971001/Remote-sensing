import os
import numpy as np
from tqdm import tqdm
from osgeo import gdal
from itertools import combinations
from utils.read_write import readtif


# 写入波段
def writetiff(im_data, im_geotrans, im_proj, path, bands):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if bands > 1:
        im_bands, im_height, im_width = im_data.shape
    elif bands == 1:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)
        dataset.SetProjection(im_proj)
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset


# 图片名列表
def file_name(root_path):
    filename = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if (os.path.splitext(file)[1] == '.jpg') or (
                    os.path.splitext(file)[1] == '.png') or (
                    os.path.splitext(file)[1] == '.tif') or (
                    os.path.splitext(file)[1] == '.TIF') or (
                    os.path.splitext(file)[1] == '.JPG'):
                filename.append(file)
    return filename


# OIF批量自动计算
def oif_multi(img_floder):
    img_list = file_name(img_floder)

    for img_name in tqdm(img_list):
        # 读取影像
        img, im_geotrans, im_proj, bands = readtif(img_floder + "/" + img_name)
        # band
        band = [i for i in range(int(bands))]
        band_combin_list = []
        for c in combinations(band, 3):
            band_combin_list.append(list(c))

        band_and_oif_list = []
        for lis in band_combin_list:
            # pearson
            img_0_1 = (img[lis[0]] - np.mean(img[lis[0]])) * (img[lis[1]] - np.mean(img[lis[1]]))
            img_0_2 = (img[lis[0]] - np.mean(img[lis[0]])) * (img[lis[2]] - np.mean(img[lis[2]]))
            img_1_2 = (img[lis[1]] - np.mean(img[lis[1]])) * (img[lis[2]] - np.mean(img[lis[2]]))

            # 分子
            img_0_1_conva = np.sum(img_0_1)
            img_0_2_conva = np.sum(img_0_2)
            img_1_2_conva = np.sum(img_1_2)

            # 标准差
            img_0 = np.std(img[lis[0]])
            img_1 = np.std(img[lis[1]])
            img_2 = np.std(img[lis[2]])

            # 分母
            img_0_1_std = img_0 * img_1
            img_0_2_std = img_0 * img_2
            img_1_2_std = img_1 * img_2

            # 相关系数
            r_0_1 = img_0_1_conva / img_0_1_std
            r_0_2 = img_0_2_conva / img_0_2_std
            r_1_2 = img_1_2_conva / img_1_2_std

            # OIF
            oif = (img_0 + img_1 + img_2) / (abs(r_0_1) + abs(r_0_2) + abs(r_1_2))

            band_and_oif = lis
            band_and_oif.append(oif)
            band_and_oif_list.append(band_and_oif)

        band_and_oif_list = np.array(band_and_oif_list)

        best_index = np.argmax(band_and_oif_list[:, 3])
        best_bands = band_and_oif_list[best_index][0:-1]
        best_bands = np.array(list(map(int, best_bands)))

        im_data = [img[best_bands[0]], img[best_bands[1]], img[best_bands[2]]]

        im_data = np.array(im_data)
        name = img_name[:-4]
        path = img_floder + "/" + 'bands_OIF'

        if not os.path.exists(path):
            os.makedirs(path)

        bands_name = name + '~' + str(best_bands[0]) + '_' + str(best_bands[1]) + '_' + str(best_bands[2])
        write_path = '{}/{}.TIF'.format(path, bands_name)
        writetiff(im_data, im_geotrans, im_proj, write_path, bands)   # 保存成多光谱tif格式


oif_multi(r'E:\RS\samples\1')
