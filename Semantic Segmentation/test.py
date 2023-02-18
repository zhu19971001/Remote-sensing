import numpy as np
import torch
import yaml
import os
from models import unet, unetplusplus
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
from tqdm import tqdm
from utils.read_write import readtif_allinfo, file_name, write


def tesT(test_folder, mean, std, model, weight_path, chanels, classnum, save_test_folder, encoder):
    image_list = file_name(test_folder)
    for image in tqdm(image_list):
        w, h, bands, img, extend, dgp = readtif_allinfo(test_folder + "\\" + image)
        img = img.swapaxes(1, 0).swapaxes(1, 2)
        datatransfoms = transforms.Compose([
            transforms.ToTensor(),      # hwc2chw
            transforms.Normalize(mean, std)
        ])
        img = np.uint8(img)
        img = datatransfoms(img)
        img = torch.unsqueeze(img, 0)

        img = torch.autograd.Variable(img.cuda())

        net = eval(model) if isinstance(model, str) else model
        # net = unet.unet(chanels, classnum)
        net.load_state_dict(torch.load(weight_path))
        net = net.cuda().eval()

        img_test = net(img)
        img = torch.argmax(img_test, dim=1)
        img = np.array(img.cpu())

        if not os.path.exists(save_test_folder):
            os.makedirs(save_test_folder)

        save_test_path = save_test_folder + "\\" + image
        write(img, extend, dgp, save_test_path)


def labelVisualize(img):
    img_out = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_out[i][j] = np.argmax(img[i][j])
    return img_out


if __name__ == '__main__':
    with open('./config/test.yaml', 'r') as file:
        opt = yaml.load(file.read(), Loader=yaml.FullLoader)

    tesT(test_folder=opt['test_folder'],
         mean=opt['mean'],
         std=opt['std'],
         model=opt['model_from_zhuguoqing'],
         weight_path=opt['weight_path'],
         chanels=opt['chanels'],
         classnum=opt['classnum'],
         save_test_folder=opt['save_test_folder'],
         encoder=opt['encoder']
         )
