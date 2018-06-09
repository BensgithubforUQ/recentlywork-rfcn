import gc
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable
import torchvision
from dataset import MyData, MyTestData
import cv2

from model import Feature_FCN, Deconv

from tensorboardX import SummaryWriter
from datetime import datetime
import os
import pdb
from myfunc import make_image_grid
import time
import math
import matplotlib.pyplot as plt

# ECSSD
train_root = '/home/wbm/桌面/未命名文件夹/RFCN-master/testdata/images'  # training dataset
val_root = '/home/wbm/桌面/未命名文件夹/RFCN-master/testdata/GT'  # validation dataset
check_root = '/home/wbm/桌面/未命名文件夹/RFCN-master/fcn_parameters'  # save checkpoint parameters
val_output_root = '/home/wbm/桌面/未命名文件夹/RFCN-master/fcn_validation/testdata'  # save validation results
bsize = 1  # batch size
iter_num = 20  # training iterations
r_num = 3  # recurrence
ptag = '/home/wbm/桌面/未命名文件夹/RFCN-master/testdata/MR'  # prior map

std = [.229, .224, .225]
mean = [.485, .456, .406]

os.system('rm -rf ./runs/*')
writer = SummaryWriter('./runs/'+datetime.now().strftime('%B%d  %H:%M:%S'))

if not os.path.exists('./runs'):
    os.mkdir('./runs')

if not os.path.exists(check_root):
    os.mkdir(check_root)

if not os.path.exists(val_output_root):
    os.mkdir(val_output_root)

# models
feature = Feature_FCN()
# feature.cuda()

deconv = Deconv()
# deconv.cuda()
feature.load_state_dict(torch.load('/home/wbm/桌面/未命名文件夹/RFCN-master/fcn_parameters/feature-epoch-0-step-360.pth'))
deconv.load_state_dict(torch.load('/home/wbm/桌面/未命名文件夹/RFCN-master/fcn_parameters/deconv-epoch-0-step-360.pth'))  
train_loader = torch.utils.data.DataLoader(
    MyData(train_root, transform=True, ptag=ptag),
    batch_size=bsize, shuffle=True, num_workers=4, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    MyTestData(val_root, transform=True, ptag=ptag),
    batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

criterion = nn.BCEWithLogitsLoss()

optimizer_deconv = torch.optim.Adam(deconv.parameters(), lr=1e-3)
optimizer_feature = torch.optim.Adam(feature.parameters(), lr=1e-4)

istep = 0


def validation(val_loader, output_root, feature, deconv):
    if not os.path.exists(output_root):
        os.mkdir(output_root)
    for ib, (data, _, img_name, img_size) in enumerate(val_loader):
        print(ib)

        # inputs = Variable(data).cuda()
        inputs = Variable(data)

        feats = feature(inputs)
        feats = feats[-3:]
        feats = feats[::-1]
        msk = deconv(feats)

        msk = functional.upsample(msk, scale_factor=4)

        msk = functional.sigmoid(msk)

        mask = msk.data[0, 0].cpu().numpy()
        mask = cv2.resize(mask, dsize=(img_size[0][0], img_size[1][0]))
        plt.imsave(os.path.join(output_root, img_name[0]+'.png'), mask, cmap='gray')


for it in range(iter_num):
    for ib, (data, _, lbl) in enumerate(train_loader):
        inputs = Variable(data)
        # inputs = Variable(data).cuda()
        # lbl = Variable(lbl.unsqueeze(1)).cuda()
        lbl = Variable(lbl.unsqueeze(1))
        loss = 0

        feats = feature(inputs)
        feats = feats[-3:]
        feats = feats[::-1]
        msk = deconv(feats)
        msk = functional.upsample(msk, scale_factor=4)
        prior = functional.sigmoid(msk)
        loss += criterion(msk, lbl)

        deconv.zero_grad()
        feature.zero_grad()

        loss.backward()

        optimizer_feature.step()
        optimizer_deconv.step()

        # visulize
        image = make_image_grid(inputs.data[:, :3], mean, std)
        writer.add_image('Image', torchvision.utils.make_grid(image), ib)
        msk = functional.sigmoid(msk)
        mask1 = msk.data  # mskdata,分割出来的。
        mask1 = mask1.repeat(1, 3, 1, 1)
        acc = math.e ** (0 - loss)
        writer.add_image('Image2', torchvision.utils.make_grid(mask1), ib)
        print('loss: %.4f,  acc %.4f, (epoch: %d, step: %d)' % (loss.data[0],  acc, it, ib))
        writer.add_scalar('M_global', loss.data[0], istep)
        writer.add_scalar('acc', acc.data[0], istep)
        istep += 1

        del inputs, msk, lbl, loss, feats, mask1, image, acc
        gc.collect()
        if ib % 30 == 0:
            filename = ('%s/deconv-epoch-%d-step-%d.pth' % (check_root, it, ib))
            torch.save(deconv.state_dict(), filename)
            filename = ('%s/feature-epoch-%d-step-%d.pth' % (check_root, it, ib))
            torch.save(feature.state_dict(), filename)
            print('save: (epoch: %d, step: %d)' % (it, ib))
    validation(val_loader, '%s/%d'%(val_output_root, it), feature, deconv)



