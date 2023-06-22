import torch
from torchvision.transforms import Compose,ToTensor,Normalize,Resize
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
import time
import cv2
from torchvision.datasets import ImageFolder

transform =Compose([Resize([128,128]),ToTensor(),
						Normalize((0.5988722 , 0.4951797,  0.44306183),
						( 0.2429065,  0.22244553, 0.22060251))])

train_loader = ImageFolder(r"D:\pycharm\face2\race\train",transform=transform)
test_loader = ImageFolder(r"D:\pycharm\face2\race\test",transform=transform)

train_loader = DataLoader(train_loader,batch_size=64,shuffle=True)
test_loader = DataLoader(test_loader,batch_size=1000,shuffle=True)

#####################################################################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#设定在gpu设备上训练
print(device)
n_epochs = 30 # 训练迭代次数
learning_rate = 0.01 # 初始学习率
momentum = 0.9 # 引入动量
random_seed = 1
torch.manual_seed(random_seed) # 设定随机种子
####################################################################################################

class Mish(nn.Module):
	def __init__(self):
		super().__init__()
	def forward(self,x):
		x=x*(torch.tanh(F.softplus(x)))
		return x

#定义神经网络结构
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()#继承父类参数
		self.mish = Mish()#实例化激活函数
		self.conv1 = nn.Conv2d(3, 30, kernel_size=3)#2d卷积操作，用于处理二维图片
		self.bn1=nn.BatchNorm2d(30)#批量归一化
		self.conv2 = nn.Conv2d(30, 60, kernel_size=3)#卷积核大小为5
		self.bn2=nn.BatchNorm2d(60)#批量归一化
		self.fc1 = nn.Linear(60*30*30, 128)#全连接层
		self.fc2 = nn.Linear(128, 5)#5分类

	def forward(self, x):#前向传播
		x = self.mish(self.bn1(F.max_pool2d(self.conv1(x), 2)))#卷积>池化>批量归一化>激活
		x = self.mish(self.bn2(F.max_pool2d(self.conv2(x), 2)))#卷积>池化>批量归一化>激活
		# print(x.shape)
		x = x.reshape(-1, 60*30*30)#改变张量形状以输入到全连接层内
		x = self.mish(self.fc1(x))
		x = self.fc2(x)
		return F.softmax(x,dim=1)#使用softmax获取每一种类的概率

network = Net()#实例化模型
network=network.to(device)#使模型用gpu训练

optimizer = optim.SGD(network.parameters(), lr=learning_rate,
					  momentum=momentum)

scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=525,gamma=0.5)

lr_list=[]#学习率
train_losses = []#训练的loss
train_counter = []#训练次数
precision=[]#准确率
x=0#训练批次次数，当做学习率的横坐标


def train(epoch):
	network.train()  # 定义网络为训练状态
	correct = 0
	global x  # 把x当成全局变量，才能记录训练迭代次数
	for batch_idx, (data, target) in enumerate(train_loader):  # 遍历数据集
		data, target = data.to(device), target.to(device)  # 将数据集转移到gpu上训练
		x += 1  # 记录迭代次数

		optimizer.zero_grad()  # 把梯度清零
		output = network(data)  # 前向传播获取输出结果
		loss = F.cross_entropy(output, target)  # 用交叉熵计算误差

		loss.backward()  # 反向计算求取梯度，更新参数
		optimizer.step()  # 更新优化器
		scheduler.step()  # 更新学习率

		pred = output.data.max(1, keepdim=True)[1]  # 函数返回每一行的最大值和其索引，我们只需要它的第二个元素，即为其最大值索引

		correct += (pred.eq(target.data.view_as(pred)).sum()).cpu()  # 计算预测的正确个数，注意，要当前数据在gpu，需转移到cpu才能添加到列表

		lr_list.append(optimizer.state_dict()["param_groups"][0]["lr"])  # 获取优化器的学习率

		print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f},Accuracy: {}/{} ({:.3f}%)\n'.format(
			epoch, batch_idx * len(data), len(train_loader.dataset),
				   100. * batch_idx / len(train_loader), loss.item(), correct, len(train_loader.dataset),
				   100. * correct / len(train_loader.dataset)))  # 打印进度，loss和准确率

		train_losses.append(loss.item())  # 记录训练损失值
		precision.append(correct / len(train_loader.dataset))
		train_counter.append((batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))  # 记录训练次数

		torch.save(network.state_dict(), './model/model_face.pth')  # 保存模型
		torch.save(optimizer.state_dict(), './model/optimizer_face.pth')  # 保存优化器

def test():
	network.eval()#设置为测试状态
	test_loss = 0
	correct = 0
	with torch.no_grad():#测试时不用记录梯度
		for data, target in test_loader:#遍历图片和标签
			data, target = data.to(device), target.to(device)#将张量放到gpu上训练
			output = network(data)#获取输出结果
			pred = output.data.max(1, keepdim=True)[1]#获取概率最大值索引
			correct += pred.eq(target.data.view_as(pred)).sum()#计算比较正确个数
	print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))#打印测试准确率

t_1=time.time()
for i in range(1,n_epochs):#训练n_epochs次
    train(i)
t_2=time.time()
t=t_2-t_1
print("所用时间：",t)#记录训练所用时间
test()#测试

fig = plt.figure()#定义画布
plt.subplot(2, 2, 1)#把画布划分为2x2大小，取第一个位置
plt.tight_layout()#自动排版，防止文字与文字，文字与图片重合
plt.plot([a * 64 for a in range(x)], lr_list, color="r")#横轴为训练的n_epochs
plt.ylabel('lreaning rate')#可视学习率变化
plt.subplot(2, 2, 2)#取第二个位置
plt.tight_layout()
plt.plot(train_counter, train_losses, color="blue")#横轴为训练的次数
plt.ylabel('loss')#可视损失值变化
plt.subplot(2, 2, 3)#取第三个位置
plt.tight_layout()
plt.plot(train_counter, precision, color="green")#横轴为训练的次数
plt.ylabel('precision')#可视准确率变化
plt.show()#展示图片



