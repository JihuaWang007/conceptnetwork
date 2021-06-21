import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pickle as pkl

from conceptnet0527small.prototype_network import *


#图片转化
def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)#截取[0,1]之间数值
    return inp

#可视化
def visualize_stn(best_model):
    with torch.no_grad():
        # Get a batch of training data
        model.load_state_dict(best_model)

        data = next(iter(test_loader))[0].to(device)#测试集中，每一个样本（输入x）逐一载入

        input_tensor = data.cpu()
        transformed_input_tensor, theta = model.stn(data)#registration

        transformed_input_tensor = transformed_input_tensor.cpu()

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor))#将若干幅图像拼成一幅图像,并转为numpy

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))

        fig = plt.figure('Figure_3', figsize=(10, 6),frameon=False)
        axarr0 = fig.add_subplot(1, 2, 1)
        axarr1 = fig.add_subplot(1, 2, 2)
        axarr0.imshow(-in_grid,cmap='gray')
        axarr0.set_title('Initial samples',fontsize=13)
        axarr0.axis('off')
        axarr1.imshow(-out_grid,cmap='gray')
        axarr1.set_title('Registered samples',fontsize=13)
        axarr1.axis('off')
        plt.subplots_adjust(top=0.9, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)


def visualize_prototype(initprototype,learnedprototype):
    with torch.no_grad():
        initprototype=torch.from_numpy(initprototype)
        learnedprototype=learnedprototype.cpu()

        in_grid = convert_image_np(
            torchvision.utils.make_grid(initprototype))#将若干幅图像拼成一幅图像,并转为numpy

        out_grid = convert_image_np(
            torchvision.utils.make_grid(learnedprototype))

        # Plot the results side-by-side
        fig=plt.figure('Figure_4',figsize=(6,6),frameon=False)

        axarr0 = fig.add_subplot(1, 2,1)
        axarr1 = fig.add_subplot(1, 2, 2)
        axarr0.imshow(-in_grid,cmap='gray')
        axarr0.set_title('Initial prototypes',fontsize=10)
        axarr0.axis('off')
        axarr1.imshow(-out_grid,cmap='gray')
        axarr1.set_title('Trained prototypes',fontsize=10)
        axarr1.axis('off')
        plt.subplots_adjust(top=0.9, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)

best_model=torch.load('./omnidata/best_model.pth')
#print('Model.state_dict:',best_model)

initprototype=XAbh[0:outsz,:,:,:]#[0,1]


if __name__ == "__main__":
    # Visualize the STN transformation on some input batch
    visualize_stn(best_model)
    plt.ioff()
    plt.show()

    # print model's state_dict
    for param_tensor in best_model:
        # 打印 key value字典,只包含卷积层和全连接层的参数
        #print(param_tensor, '\t', best_model[param_tensor].size())
        if param_tensor=='partclassifier.weight':
            Theta1=best_model[param_tensor]
        if param_tensor == 'classifier.weight':
            Theta2=best_model[param_tensor]
    learnedprototype=torch.mm(Theta2[:,0:-512],Theta1).view(-1,C,H,W)
    #torch.clamp(learnedprototype,min=,max=)
    #learnedprototype=torch.where(learnedprototype>0.3,1,0)
    lp=learnedprototype.numpy()
    lp[lp<0.1]=0
    lp[lp>0.1]=255
    learnedprototype=torch.from_numpy(lp)
    #Visualize prototype
    visualize_prototype(initprototype, learnedprototype)
    plt.ioff()
    plt.show()

