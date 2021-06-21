import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
import gzip
import pickle as pkl
import time

#def __call__(self, *args)方法接受一定数量的变量作为输入;改变实例的内部成员的值;有返回值
#包含需要学习参数的模块都要写入__init__中
class Net(nn.Module):
    def __init__(self,Theta1,Theta2):
        super(Net, self).__init__()
        self.features = models.vgg16_bn(pretrained=True).features  # 用已经训练好的vgg16提取特征,输出为512个通道
        # print('vggfeatures:',self.features)
        # 输入为灰度图，通道数为1，因此需要把VGG16中的第一层卷积更改如下：二维卷积：输入通道=1，输出通道=64
        self.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.features = self.features.eval()  # 一定要有这行，不然运算速度会变慢（要求梯度）而且会影响结果
        # VGG16输出如下：
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))  #平均池化：输出每一个通道的平均值：[1,1,512]

        self.Theta1=Theta1#inputsize=105*105=11025, hiddensize=part number
        self.Theta2=Theta2#hiddensize=part number, outsize=class number

        hiddensz,inputsz=Theta1.shape# self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.partclassifier = nn.Linear(inputsz, hiddensz)  # 全连接：y=xA+b,
        self.partclassifier.weight.data.copy_(Theta1)
        #self.partclassifier.weight.requires_grad=False#不可学习
        self.partclassifier.bias.data.zero_()

        outsz,hidsz = Theta2.shape
        self.classifier = nn.Linear(hiddensz+512,outsz)
        self.classifier.weight.data[:,0:-512].copy_(Theta2)
        #self.classifier.weight.requires_grad=False
        self.classifier.bias.data.zero_()

        #convolution for spatial transformer:slightly non-rigid translating,rotating,scaling
        #保证输入输出的H和W不变！！
        """self.nonrigidtf = nn.Sequential(
            nn.Conv2d(1, 40, kernel_size=5),  # 输入通道为1，输出为40,net.conv2d.weight=[1,6,5,5]
            nn.AdaptiveMaxPool2d(output_size=(105, 105)),
            nn.ReLU(True),
            nn.Conv2d(40, 1, kernel_size=5),  # 输入通道为20，输出通道为1
            nn.AdaptiveMaxPool2d(output_size=(105, 105)),
            nn.ReLU(True)
        )"""

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),#输入通道为1，输出为6,net.conv2d.weight=[1,6,5,5]
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(6,10, kernel_size=5),#输入通道为6，输出通道为10
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))#平均池化：通道为10，通道数前后不变
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10, 32),#全连接
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)#全连接
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function:输入为batchsize*C*H*W
    #__call___: 使得类对象具有类似函数的功能
    #def __call__(self):
        #print('i can be called like a function')

    def stn(self, x):
        xs = self.localization(x)#x[batchsize,1,105,105],输出xs[batchsize,10,1,1]
        xs = xs.view(-1, xs.size()[1] * xs.size()[2] * xs.size()[3])#[batchsize,10]
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)#重构shape:[batchsize,6]

        # affine_grid输入是仿射矩阵(Nx2x3)和
        # the target output image size. (N×C×H×W for 2D or N×C×D×H×W for 3D)
        # 输出Tensor的尺寸(Tensor.Size(NxHxWx2))，输出的是归一化的二维网格。

        grid = F.affine_grid(theta, x.size())
        # grid_sample函数中将图像坐标归一化到[−1,1]，其中0对应-1，width-1对应1。
        x = F.grid_sample(x, grid)
        return x, theta

    def forward(self, x):
        # transform the input:
        stx0,stweight = self.stn(x) #x[batchsize,C,H,W]转换为batchsize=m*inputsize
        #stx0=self.nonrigidtf(x)
        #stx0=x
        #stweight=x

        xvgg0 = self.features(stx0)  # 得到vgg16特征
        xvgg = self.avgpool(xvgg0)  # 平均池化：[1,1,512]
        xvgg = torch.flatten(xvgg, 1)  # axis=1，按照行拼接进行拼接:512

        X = stx0.view(-1, stx0.size()[1] * stx0.size()[2] * stx0.size()[3])
        # 开始定义三层全连接网络：以隐层即part层作为目标计算损失函数
        z2=self.partclassifier(X)
        z2m,z2id=z2.max(1)#按行取最大值
        z2 -=z2m.view(-1,1)#节点的分布为exp(1-z1)
        z20 = torch.exp(z2)

        # p2=F.log_softmax(z2, dim=1)#如果以隐层作为目标约束话，则。。。

        #########把part[hiddensz]+vgg_features[512]
        z20v=torch.cat((z20,xvgg),1)#列维度数不变情况下（axis=1),拼接

        z3=self.classifier(z20v)
        #print('z3:', z3.shape)
        z3m, z3id = z3.max(1)
        z3 -= z3m.view(-1,1)
        # z30 = softmax(z3)
        output = F.log_softmax(z3, dim=1)  # log(softmax(x));dim=1是对每一行的所有元素进行运算，并使得每一行所有元素和为1。

        return output,stx0,z20#输出，几何变换后图像，最后三层全连接的中间层


#Softmax函数,z为m行p列
def softmax(x):
    """ softmax function """
    # assert(len(x.shape) > 1, "dimension must be larger than 1",矩阵)
    # print(np.max(x, axis = 1, keepdims = True)) # axis = 1, 行
    #x -= np.max(x, axis=1, keepdims=True)  # 为了稳定地计算softmax概率， 一般会减掉最大的那个元素
    #print("减去行最大值 ：\n", x)
    x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    return x

# Softmax型函数导数:一般不用
def softmaxGradient(z):
    #i=j
    g = softmax(z) * (1 - softmax(z))
    #i!=j
    #g=-softmax(z)*softmax(z)
    return g

#crd=os.path.abspath(os.path.dirname(os.getcwd()))
#img_dir = crd+'/conceptnet20210421/MNIST'#root='./data/FashionMNIST',
#权重为第一批数据,且不更新 # 每个笔画一张图像:笔画及其对应部件、类别
with open('./omnidata/XC1.txt', 'rb') as f11:
    fcXCbh = pkl.load(f11)  # 笔画训练集
with open('./omnidata/ys1.txt', 'rb') as f12:
    fcysbh = pkl.load(f12)  # 笔画训练集标签，列向量：m*num_classes
with open('./omnidata/ysp1.txt', 'rb') as f13:
    fcyspbh = pkl.load(f13) #笔画/part层标签

# 第一批part或笔画数据数据
M,C,H,W  = fcXCbh.shape# m为样本量，也就是笔画量，每一批数量不定
inputsz=H*W
hiddensz = fcyspbh.shape[0]#一个样本对应一个笔画！！注意：每一批的笔画数目不一样
#print('hidden:',hiddensize)#.shape=(271,)是一个tuple
outsz = np.max(fcysbh)+1#多个笔画对应一个类别：0，1，2，。。。
outsz=outsz.astype(np.int)#类别数量

####################权重初始化
fcX = fcXCbh.reshape(-1,C*H*W)#第一批或某一批样本作为template:输入
yp = fcyspbh#隐层
y = fcysbh#输出
Theta10 = np.zeros((hiddensz, inputsz))#输入到隐层的权重
for mi in range(M):
    id = yp[mi].astype(np.int)# 正常每行只能有一个“1”,对应每个笔画part
    Theta10[id, :] = fcX[mi, :]
Theta20 = np.zeros((outsz, hiddensz))#隐层到输出全的权重
for cls in range(outsz):
    id2 = np.array(np.where(y == cls))  # 对应一个类别的所有索引
    for j2 in id2:
        Theta20[cls, j2] = 1
Theta1t=torch.from_numpy(Theta10)
Theta2t=torch.from_numpy(Theta20)

###images训练集
#with open('./omnidata/XA.txt', 'rb') as f2:
    #XAbh = pkl.load(f2)  # 训练集
with open('./omnidata/XAreged.txt', 'rb') as f21:#XAreged.txt
   XAbh=pkl.load(f21)
with open('./omnidata/y.txt', 'rb') as f3:
   ybh = pkl.load(f3)  # 训练集标签，列向量：m行1列：num_classes
XAbh=XAbh#/255
m,C0,H0,W0  = XAbh.shape
tm=np.int(m*0.7)
##############dataset:
X_train=XAbh[:tm,:,:,:]
y_train = ybh[:tm]
X_test=XAbh[tm:,:,:,:]
y_test = ybh[tm:]
X_train_tensor = torch.from_numpy(X_train).to(torch.float32).view(-1,C,H,W)#*(1/255)
X_test_tensor = torch.from_numpy(X_test).to(torch.float32).view(-1,C,H,W)#*(1/255)
y_train_tensor = torch.from_numpy(y_train).to(torch.long)#.view(-1,1)#
y_test_tensor = torch.from_numpy(y_test).to(torch.long)#.view(-1,1)

mnist_train = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
mnist_test = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
# Training and testing dataset
batch_size = 96
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

#设置优化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net(Theta1=Theta1t,Theta2=Theta2t).to(device)#把tensor放在设备上

#optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

#训练、测试
def train(epoch):
   model.train()
   correct = 0
   hidw=[]
   outw=[]
   for batch_idx, (data, target) in enumerate(train_loader):
       print('batch_index=',batch_idx)
       print('data:',data.shape)
       data, target = data.to(device), target.to(device)

       optimizer.zero_grad()
       with torch.set_grad_enabled(True):
           output, stx0, z20 = model(data)

       loss = F.nll_loss(output, target, reduction='sum')  # target给的是索引值或标签号，nll_loss则是预测结果对应索引值处的均值。
       # 使用nn.CrossEntropyLoss时，label必须是[0, #classes] 区间的一个数字 ，
       # 而不可以是one-hot encoded 目标向量

       pred = output.max(1, keepdim=True)[1]  # 返回每一行中最大的元素并返回索引，返回了两个数组，[1]则为索引列
       correct += pred.eq(target.view_as(pred)).sum().item()  # view_as=self.view(tensor.size())，累加每批样本中相等的数目
       # .sum()=tensor(5,),而.item()用于取出tensor中的值。
       # reshape ≈ tensor.contiguous().view；
       loss.backward()
       optimizer.step()

   # 所有batches运算结束：
   accuracy = correct / len(train_loader.dataset)  # 总样本量

   print('Train Epoch: {} \nLoss: {:.6f} \tAccuracy: {}/{}({:.2f}%)'.format(
        epoch, loss.item(), correct, len(train_loader.dataset), accuracy*100.))

   torch.save(model.state_dict(), './omnidata/best_model.pth',_use_new_zipfile_serialization=False)#包含卷积层和全连接层的参数

   return loss, accuracy

def test():
    with torch.no_grad():
        #best_model = torch.load('./omnidata/best_model.pth',map_location='cpu')
        #model.load_state_dict(best_model)
        model.eval()
        test_loss = 0
        correct = 0
        #print('test_loader:',enumerate(test_loader))
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, stx0, z20 = model(data)

            # sum up batch loss
            test_loss += F.nll_loss(output, target,  reduction='sum').item() #size_average=False
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = correct / len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
              .format(test_loss, correct, len(test_loader.dataset), 100. * accuracy))


if __name__ == "__main__":
    trainstart=time.time()
    for epoch in range(1, 30):
        print('epoch:', epoch)
        loss, accuracy = train(epoch)

        teststart = time.time()
        test()
        testtime = time.time() - teststart
        print('Testing in {:.0f}s'.format(testtime))

    traintime=time.time()-trainstart
    print('Training complete in {:.0f}m {:.0f}s'.format(traintime // 60, traintime % 60))

