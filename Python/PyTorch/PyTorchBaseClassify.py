'''
- 使用torchvision加载和规范化CIFAR10训练和测试数据集
- 定义一个卷积神经网络
- 定义一个loss函数
- 使用训练数据来训练网络
- 使用测试数据来测试网络
'''

# -------加载数据
import torch
import torchvision
import torchvision.transforms as transforms
transform = transforms.Compose(
	[transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
										download=True, transform=transform)
# num_workers: 用多少个子进程加载数据。0表示数据将在主进程中加载
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
										shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, 
										download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
										shuffle=False, num_workers=0)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
			'dog', 'frog', 'horse', 'ship', 'truck')


# # -------可视化数据集中的图片
import matplotlib.pyplot as plt
import numpy as np
def imshow(img):
	# 反标准化
	img = img / 2 + 0.5
	# tensor转numpy
	npimg = img.numpy()
	# 转置操作
	# x,y,z对应为(0, 1, 2)
	# 这里是(1, 2, 0),表示x变为y,y变为z,z变x
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()

# # 取得迭代器
# dataiter = iter(trainloader)
# # 取得迭代器中的数据
# images, labels = dataiter.next()
# # images[0].size()为[3, 32, 32]表示3通道，大小为32*32
# # 把数据转为网格图片
# imshow(torchvision.utils.make_grid(images))
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# -------定义一个卷积神经网络
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(16*5*5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16*5*5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x
net = Net()

# -------定义损失函数以及优化的梯度下降
import torch.optim as optim
# 交叉熵loss
# 分类问题常用的损失函数为交叉熵
# 交叉熵描述了两个概率分布之间的距离，当交叉熵越小说明二者之间越接近
criterion = nn.CrossEntropyLoss()
# 带动量的SGD。monentum为动量参数
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# ------训练网络
for epoch in range(2):
	running_loss = 0.0
	for i, data in enumerate(trainloader, 0):
		# 在读入数据时，minibatch为4，所以这里的inputs和lables也是4
		inputs, labels = data
		optimizer.zero_grad()
		# 因为inputs为4，故outputs为4个长度为10的张量
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		running_loss += loss.item()
		if i%2000 == 1999:
			print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
			running_loss = 0.0
print('Finished Training')

# # -------单个数据测试网络
# dataiter = iter(testloader)
# images, labels = dataiter.next()
# imshow(torchvision.utils.make_grid(images))
# # 正确的分类结果
# print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
# outputs = net(images)
# # 输出张量的最大值，第二个参数表示指定的维度，设置为1时表示与输入形状保持一致
# _, predicted = torch.max(outputs, 1)
# print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

# ------全部测试集
correct = 0
total = 0
with torch.no_grad():
	for data in testloader:
		images, labels = data
		outputs = net(images)
		_, predicted = torch.max(outputs, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# -------查看那些类别不能很好识别
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
	for data in testloader:
		images, labels = data
		outputs = net(images)
		_, predicted = torch.max(outputs, 1)
		c = (predicted == labels).squeeze()
		for i in range(4):
			label = labels[i]
			class_correct[label] += c[i].item()
			class_total[label] += 1
for i in range(10):
	print('Accuracy of %5s : %2d %%' % ( classes[i], 100 * class_correct[i] / class_total[i]))

# # 把网络放到GPU上加速
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
# net.to(device)
# inputs, labels = inputs.to(device), labels.to(device)