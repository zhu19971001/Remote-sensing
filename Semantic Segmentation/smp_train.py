import random
import time
import warnings
import numpy as np
import torch
import yaml
import torch.nn as nn
from tqdm import tqdm
from smp_dataprocess_new import MyDataset
from torch.autograd import Variable
from models.smp_model import smp_models
from models import unet, unetplusplus


warnings.filterwarnings('ignore')
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(2022)


def train(classNum, EPOCHES,size, BATCH_SIZE,channels, numworks, train_mean, train_std, val_mean, val_std, optimizer_name,
          model_save_path, train_img_path, train_lab_path, val_img_path, val_lab_path, train_lab_folder, val_lab_folder, model, encoder, activate, loss,
          ndvi, exg, vdvi, rvi, ndgi, ngbdi, ndcsi):

    train_dataset = MyDataset(train_img_path, train_lab_path, train_lab_folder, train_mean, train_std, ndvi, exg, vdvi, rvi, ndgi, ngbdi, ndcsi)
    val_dataset = MyDataset(val_img_path, val_lab_path, val_lab_folder, val_mean, val_std, ndvi, exg, vdvi, rvi, ndgi, ngbdi, ndcsi)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=numworks)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=numworks)

    spm_model = smp_models(encoder, channels, classNum, activate)  # 实例化
    model = eval(model) if isinstance(model, str) else model
    # model = spm_model.unet()
    model = model.cuda()

    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, weight_decay=1e-3, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2,  T_mult=2, eta_min=1e-5)

    citerion = nn.BCELoss()

    train_loss_list, val_loss_list, train_acc_list, val_acc_list = [], [], [], []
    for epoch in range(EPOCHES):
        print('epoch:{}/{}'.format(epoch + 1, EPOCHES))
        train_loss = 0.0
        train_acc = 0.0
        start_time = time.time()

        model.train()

        for i, (img_data, label, lab_origin) in enumerate(train_data_loader):
            if img_data.shape[0] < BATCH_SIZE:
                continue

            img_data = Variable(img_data.cuda())  # .long()
            label = Variable(label.cuda())  # .long()
            lab_origin = torch.tensor(lab_origin).cuda()
            optimizer.zero_grad()

            # forward
            output = model(img_data)
            loss = citerion(output, label)
            train_loss += loss.item()
            pred = torch.argmax(output, dim=1).type(torch.uint8)
            train_acc += torch.sum(pred == lab_origin) / (BATCH_SIZE * size**2)

            loss.backward()
            optimizer.step()

        # scheduler.step() # 余弦退火

        train_loss = train_loss / len(train_data_loader)
        train_acc = train_acc / len(train_data_loader)
        print('train_loss:{}, train_acc:{}'.format(train_loss, train_acc))

        # 验证集
        val_loss, val_acc = val(model, citerion, val_data_loader, BATCH_SIZE, size)

        scheduler.step(val_loss)                   # 调整学习率

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

        path = model_save_path + '{}.pth'.format(epoch + 1)
        torch.save(model.state_dict(), path)

        print("epoch_time:{}".format(time.time() - start_time))

    return train_loss_list, train_acc_list, val_loss_list, val_acc_list


def val(net, criterion, val_data_loader, BATCH_SIZE, size):
    net = net.eval()

    val_loss = 0.0
    val_acc = 0.0

    with torch.no_grad():
        for i, (img_data, label, lab_origin) in enumerate(val_data_loader):
            if img_data.shape[0] < BATCH_SIZE:
                continue
            img_data = Variable(img_data.cuda())
            label = Variable(label.cuda())
            lab_origin = torch.tensor(lab_origin).cuda()

            # forward
            output = net(img_data)
            loss = criterion(output, label)
            val_loss += loss.item()
            pred = torch.argmax(output, dim=1).type(torch.uint8)
            val_acc += torch.sum(pred == lab_origin) / (BATCH_SIZE*size**2)

        # 计算准确率，计算损失
        val_loss = val_loss / len(val_data_loader)
        val_acc = val_acc / len(val_data_loader)

        print('val_loss:{}, val_acc:{}'.format(val_loss, val_acc))
    return val_loss, val_acc


if __name__ == '__main__':
    with open('./config/train.yaml', 'r') as file:
        opt = yaml.load(file.read(), Loader=yaml.FullLoader)

    train_loss, train_acc, val_loss, val_acc = train(classNum=opt['classnum'],
                                                     EPOCHES=opt['epochs'],
                                                     size=opt['size'],
                                                     BATCH_SIZE=opt['batchsize'],
                                                     channels=opt['channels'],
                                                     numworks=opt['workers'],
                                                     train_mean=opt['train_mean'],
                                                     train_std=opt['train_std'],
                                                     val_mean=opt['val_mean'],
                                                     val_std=opt['val_std'],
                                                     optimizer_name=opt['optimizier'],
                                                     model_save_path=opt['weights_save_path'],
                                                     train_img_path=opt['train_img_path'],
                                                     train_lab_path=opt['train_lab_path'],
                                                     val_img_path=opt['val_img_path'],
                                                     val_lab_path=opt['val_lab_path'],
                                                     train_lab_folder=opt['train_lab_folder'],
                                                     val_lab_folder=opt['val_lab_folder'],
                                                     model=opt['model'],
                                                     encoder=opt['encoder'],
                                                     activate=opt['activate'],
                                                     loss=opt['loss'],
                                                     ndvi=opt['ndvi'],
                                                     exg=opt['exg'],
                                                     vdvi=opt['vdvi'],
                                                     rvi=opt['rvi'],
                                                     ndgi=opt['ndgi'],
                                                     ngbdi=opt['ngbdi'],
                                                     ndcsi=opt['ndcsi'])

    train_acc = np.array(torch.stack(train_acc).cpu())
    val_acc = np.array(torch.stack(val_acc).cpu())

    if True:
        import matplotlib.pyplot as plt
        EPOCHES = opt['epochs']
        epochs = range(1, EPOCHES + 1)

        plt.plot(epochs, train_loss, color='r', label='train')
        plt.plot(epochs, val_loss, color='b', label='validation')
        plt.plot(xlabel='epoch')
        plt.plot(ylabel='loss')
        plt.title('Training and validation loss')
        plt.legend(loc='best')
        plt.savefig('train_val_loss.png')
        plt.show()

        plt.plot(epochs, train_acc, color='r', label='train')
        plt.plot(epochs, val_acc, color='b', label='validation')
        plt.plot(xlabel='epoch')
        plt.plot(ylabel='accuracy')
        plt.title('Training and validation accuracy')
        plt.legend(loc='best')
        plt.savefig('train_val_accuracy.png')
        plt.show()
