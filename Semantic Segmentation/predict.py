import torch
import yaml
from utils.read_write import readtif_allinfo, write
from models import unet, unetplusplus
from models.smp_model import smp_models
import numpy as np
from utils.sth_for_predict import TifCroppingArray, Result


def predict(imgpath, SideLength, crop_size, chanels, classnum, weight_path, save_path, model, encoder, activate):

    # net = unet.unet(chanels, classnum)
    spm_model = smp_models(encoder, chanels, classnum, activate)  # 实例化
    net = eval(model) if isinstance(model, str) else model
    net.load_state_dict(torch.load(weight_path))
    net = net.cuda().eval()

    w, h, bands, img, im_geotrans, im_proj = readtif_allinfo(imgpath)
    shape = (h, w)
    img = np.uint8(img)
    img = img / 255   # 归一化
    img = img.swapaxes(1, 0).swapaxes(1, 2)   # chw2hwc

    TifArray, RowOver, ColumnOver = TifCroppingArray(img, SideLength, crop_size)
    pre_result = []
    for i in range(len(TifArray)):
        for j in range(len(TifArray[0])):
            img = TifArray[i][j]
            mean, std = mean_std(img, bands)
            img = (img - mean) / std            # 标准化
            img = img.swapaxes(1,  2).swapaxes(1, 0)      # hwc2chw
            img = torch.tensor(img)
            img = torch.unsqueeze(img, 0)     # 1, 6, 512, 512
            img = torch.autograd.Variable(img.cuda()).type(torch.float)
            pred = net(img)
            pred = torch.argmax(pred, dim=1)
            pre_result.append(pred)

    pre_result = torch.stack(pre_result)
    pre_result = np.array(pre_result.cpu(), dtype=np.uint8)
    result = Result(pre_result, shape, TifArray, SideLength, RowOver, ColumnOver, crop_size)
    write(result, im_geotrans, im_proj, save_path)


def mean_std(img, bands):
    mean = []
    std = []
    for i in range(bands):
        mean_i = np.mean(img[:, :, i])
        var_i = np.mean((img[:, :, i] - mean_i) ** 2)
        var_i = np.sqrt(var_i)
        mean.append(mean_i)
        std.append(var_i)
    mean = np.array(mean)
    std = np.array(std)
    return mean, std


if __name__ == '__main__':
    with open('./config/predict.yaml', 'r') as file:
        opt = yaml.load(file.read(), Loader=yaml.FullLoader)

    predict(imgpath=opt['imgpath'],
            SideLength=opt['SideLength'],
            crop_size=opt['crop_size'],
            chanels=opt['chanels'],
            classnum=opt['classnum'],
            weight_path=opt['weight_path'],
            save_path=opt['save_path'],
            model=opt['model'],
            encoder=opt['encoder'],
            activate=opt['activate'])
