import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
#import matplotlib as mpl
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

def visualize_prototype(learnedprototype,genexample):
    with torch.no_grad():
        learnedprototype=learnedprototype.cpu()

        out_grid = convert_image_np(
            torchvision.utils.make_grid(learnedprototype))

        example = convert_image_np(
            torchvision.utils.make_grid(genexample))

        # Plot the results side-by-side
        fig=plt.figure('Prototype',figsize=(6,6),frameon=False)

        axarr1 = fig.add_subplot(1, 2, 1)
        axarr2 = fig.add_subplot(1, 2, 2)

        axarr1.imshow(out_grid)
        axarr1.set_title('Trained prototype',fontsize=12)
        axarr1.axis('off')

        axarr2.imshow(example)
        axarr2.set_title('Generated example', fontsize=12)
        axarr2.axis('off')

        plt.subplots_adjust(top=0.9, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)


best_model=torch.load('./omnidata/best_model.pth',map_location=lambda storage, loc: storage)#,map_location='cpu')
#print('Model.state_dict:',best_model)
model.load_state_dict(best_model)

initprototype=XAbh[0:outsz,:,:,:]#[0,1]
inity=ybh[0:outsz]

genexample=torch.zeros(size=initprototype.shape)
#initexample=np.zeros((initprototype.shape))
m,C,H,W=initprototype.shape
grid=np.zeros((m,H,W,2))


if __name__ == "__main__":

    # print model's state_dict
    for param_tensor in best_model:
        # 打印 key value字典,只包含卷积层和全连接层的参数
        #print(param_tensor, '\t', best_model[param_tensor].size())
        if param_tensor=='partclassifier.weight':
            Theta1=best_model[param_tensor]
        if param_tensor == 'classifier.weight':
            Theta2=best_model[param_tensor]
    learnedprototype=torch.mm(Theta2[:,0:-512],Theta1).view(-1,C,H,W)#
    lp = learnedprototype.numpy()
    lp[lp < 0.1] = 0
    lp[lp > 0.1] = 255
    learnedprototype = torch.from_numpy(lp)

    """one method
    #依据学习好的原型或模板生成：模板上像素值不为0的点（笔画轨迹）随机摆动一定范围，如两个像素范围
    ind=torch.nonzero(learnedprototype>0.99)#取出非零元素索引，为tensor([b,0,h,w],[],...)
    #h、w加上随机摆动
    leng=ind.shape[0]
    turb1=torch.from_numpy(np.random.randint(0,1,size=leng))
    turb2 = torch.from_numpy(np.random.randint(0, 1, size=leng))
    print(turb1)
    ind[:,2]=ind[:,2]+turb1.long()
    ind[:,3] = ind[:, 3] + turb2.long()
    print(ind)
    indnp=ind.numpy()
    genexamplenp=genexample.numpy()
    genexamplenp[tuple(indnp.T)]=255
    plt.imshow(genexamplenp[0, 0, :, :])
    plt.show()
    genexample1=torch.from_numpy(genexamplenp)

    #初始模板
    indit = np.argwhere(initprototype)  # 取出非零元素索引，为tensor([b,0,h,w],[],...)
    print(indit[-1,:])
    lengit = indit.shape[0]
    turb1it = np.random.randint(0, 2, size=lengit)
    turb2it = np.random.randint(0, 2, size=lengit)
    print(turb1it[-1],turb2it[-1])
    indit[:, 2] = indit[:, 2] + turb1it
    indit[:, 3] = indit[:, 3] + turb2it
    print(indit)
    initexample[tuple(indit.T)] = 255
    """
    ##########another method:
    h=np.arange(H)
    w=np.arange(W)
    meshh=(2.0*h-H)/H#[-1,1]
    meshw = (2.0 * w - W) / W  # [-1,1]
    for mi in range(m):
        meshhrand = np.random.randn(H) / 50 + meshh
        #plt.plot(meshh,'r',meshhrand,'g')
        #plt.show()

        meshwrand = np.random.randn(W) / 50 + meshw
        hx, wy = np.meshgrid(meshhrand, meshwrand)
        # print(hx,'-----',wy)
        grid[mi,:,:,0]=hx
        grid[mi,:,:,1]=wy
    gridt=torch.from_numpy(grid)
    genexample=F.grid_sample(learnedprototype,gridt.float())


    #Visualize prototype
    visualize_prototype(learnedprototype, genexample)
    plt.ioff()
    plt.show()

    """
    y_test1 = inity
    X_test_tensor1 = genexample.float()#learnedprototype #genexample1.float()
    y_test_tensor1 = torch.from_numpy(y_test1).to(torch.long)  # .view(-1,1)
    mnist_test1 = torch.utils.data.TensorDataset(X_test_tensor1, y_test_tensor1)
    # testing dataset
    batch_size = 16
    test_loader = torch.utils.data.DataLoader(mnist_test1, batch_size=batch_size, shuffle=False)

    #test
    teststart = time.time()
    test()
    testtime = time.time() - teststart
    print('Testing in {:.0f}s'.format(testtime))"""
