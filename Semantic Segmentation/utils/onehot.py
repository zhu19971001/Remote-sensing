import cv2
import os
import numpy as np


def color_dict(labelFolder, classNum):
    colorDict = []
    #  获取文件夹内的文件名
    ImageNameList = os.listdir(labelFolder)
    for i in range(len(ImageNameList)):
        ImagePath = labelFolder + "/" + ImageNameList[i]
        img = cv2.imread(ImagePath).astype(np.uint32)
        #  如果是灰度，转成RGB
        if(len(img.shape) == 2):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.uint32)
        img_new = img[:, :, 0] * 1000000 + img[:, :, 1] * 1000 + img[:, :, 2]
        unique = np.unique(img_new)

        for j in range(unique.shape[0]):
            colorDict.append(unique[j])
        colorDict = sorted(set(colorDict))
        if(len(colorDict) == classNum):
            break

    colorDict_RGB = []
    for k in range(len(colorDict)):
        color = str(colorDict[k]).rjust(9, '0')

        color_RGB = [int(color[0 : 3]), int(color[3 : 6]), int(color[6 : 9])]
        colorDict_RGB.append(color_RGB)

    colorDict_RGB = np.array(colorDict_RGB)
    colorDict_GRAY = colorDict_RGB.reshape((colorDict_RGB.shape[0], 1, colorDict_RGB.shape[1])).astype(np.uint8)
    colorDict_GRAY = cv2.cvtColor(colorDict_GRAY, cv2.COLOR_BGR2GRAY)
    return colorDict_GRAY


def pixel_to_index(label, labelfolder, classnum):
    colodictgray = color_dict(labelfolder, classnum)
    for i in range(colodictgray.shape[0]):
        label[label == colodictgray[i][0]] = i
    return label


def one_hot(label, labelFolder, classNum):
    colorDict_GRAY = color_dict(labelFolder, classNum)
    for i in range(colorDict_GRAY.shape[0]):
        label[label == colorDict_GRAY[i][0]] = i
    new_label = np.zeros(label.shape + (classNum,))
    for i in range(classNum):
        new_label[label == i, i] = 1
    label = new_label
    return label