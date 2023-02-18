import numpy as np
from itertools import combinations


def oif_choose(img, bands, choices):

    band = [i for i in range(bands)]
    band_combin_list = []
    for c in combinations(band, choices):
        band_combin_list.append(list(c))  # band组合

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
    return im_data
