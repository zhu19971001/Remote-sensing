import os
import numpy as np
from tqdm import tqdm
import argparse
from utils.read_write import readtif


def input_args():
    parser = argparse.ArgumentParser(description="calculating mean and std")
    parser.add_argument("--inputdir", type=str,
                        default=r'E:\RS\dataset\image\val')
    return parser.parse_args()


if __name__ == "__main__":
    opt = input_args()
    img_files = []
    img_dir = opt.inputdir
    files = os.listdir(img_dir)
    img_files.extend([os.path.join(img_dir, file) for file in files])

    mean = np.asarray([0, 0, 0, 0, 0, 0], dtype=np.float64)
    var = np.asarray([0, 0, 0, 0, 0, 0], dtype=np.float64)

    for img_file in tqdm(img_files, desc="calculating mean", mininterval=0.1):
        img = readtif(img_file)
        img = img / 255
        for i in range(len(img)):
            mean[i] += np.mean(img[i])
    mean = mean / len(img_files)

    for img_file in tqdm(img_files, desc="calculating var", mininterval=0.1):
        img = readtif(img_file)
        img = img / 255
        for i in range(len(img)):
            var[i] += np.mean((img[i] - mean[i])**2)
    for i in range(len(var)):
        var[i] = var[i] / len(img_files)
        var[i] = np.sqrt(var[i])

    print("mean:{}".format(mean))
    print("std:{}".format(var))
