from osgeo import gdal
from osgeo import osr
import numpy as np
import os

def file_name(root_path):
    filename = []
    # 照片遍历
    for file in os.listdir(root_path):
        if (os.path.splitext(file)[1] == '.jpg') or (
                os.path.splitext(file)[1] == '.png') or (
                os.path.splitext(file)[1] == '.tif')  or (
                os.path.splitext(file)[1] == '.TIF') or (
                os.path.splitext(file)[1] == '.JPG'):
            filename.append(file)
    return filename

def readtif(tif_path):
    if tif_path.endswith('.tif') or tif_path.endswith('.TIF'):
        dataset = gdal.Open(tif_path)
        pcs = osr.SpatialReference()
        dgp = dataset.GetProjection()
        pcs.ImportFromWkt(dgp)
        gcs = pcs.CloneGeogCS()
        bands = dataset.RasterCount
        extend = dataset.GetGeoTransform()
        w = dataset.RasterXSize
        h = dataset.RasterYSize
        shape = (dataset.RasterYSize, dataset.RasterXSize)
    else:
        raise "Unsupported file format"
    if bands > 1:
        img = dataset.ReadAsArray(0, 0, w, h)  # (height, width)
    else:
        img = dataset.GetRasterBand(1).ReadAsArray()

    return img


def readtif_allinfo(tif_path):
    if tif_path.endswith('.tif') or tif_path.endswith('.TIF'):
        dataset = gdal.Open(tif_path)
        pcs = osr.SpatialReference()
        dgp = dataset.GetProjection()
        pcs.ImportFromWkt(dgp)
        gcs = pcs.CloneGeogCS()
        bands = dataset.RasterCount
        extend = dataset.GetGeoTransform()
        w = dataset.RasterXSize
        h = dataset.RasterYSize
        shape = (dataset.RasterYSize, dataset.RasterXSize)
    else:
        raise "Unsupported file format"

    if bands > 1:
        img = dataset.ReadAsArray(0, 0, w, h)  # (height, width)
    else:
        img = dataset.GetRasterBand(1).ReadAsArray()

    return w, h, bands, img, extend, dgp


def write(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if dataset != None:
        dataset.SetGeoTransform(im_geotrans)
        dataset.SetProjection(im_proj)
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset


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
    if dataset != None:
        dataset.SetGeoTransform(im_geotrans)
        dataset.SetProjection(im_proj)
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset

