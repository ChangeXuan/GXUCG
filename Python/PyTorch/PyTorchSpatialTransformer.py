# ------------空间转换器
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader = torch.utils.data.DataLoader(
		datasets.MNIST(root='./data/data', train=True, download=True,
			transform=transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,))
		])), batch_size=64, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(
		datasets.MNIST(root='./data/data', train=False, transform=transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,))
		])), batch_size=64, shuffle=True, num_workers=0)

# -------空间转换器大概分为三个组成部分
'''
- 局部网络：局部化网络是对变换参数进行回归的常规CNN
- 网格生成器：在输入图像中生成对应于输出图像中的每个像素的坐标网格
- 采样器：使用转换的参数并将其应用于输入图像
'''
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
		# 随机丢弃
		# 随机归零整个通道
		self.conv2_drop = nn.Dropout2d()
		self.fc1 = nn.Linear(320, 50)
		self.fc2 = nn.Linear(50, 10)
		# 构建局部网络
		# 论文中说，局部网络可以是全连接层也可以是卷积层
		# 但是最后一层一定是回归层，用来输出参数
		self.localization = nn.Sequential(
			nn.Conv2d(1, 8, kernel_size=7),
			nn.MaxPool2d(2, stride=2),
			nn.ReLU(True),
			nn.Conv2d(8, 10, kernel_size=5),
			nn.MaxPool2d(2, stride=2),
			nn.ReLU(True)
		)
		# 构建3*2仿射矩阵的回归网络
		self.fc_loc = nn.Sequential(
			nn.Linear(10*3*3, 32),
			nn.ReLU(True),
			nn.Linear(32, 3*2)
		)
		# 用恒等变换初始化权重
		self.fc_loc[2].weight.data.zero_()
		self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

	def stn(self, x):
		xs = self.localization(x)
		xs = xs.view(-1, 10*3*3)
		theta = self.fc_loc(xs)
		theta = theta.view(-1, 2, 3)
		# 使用theta构建网格生成器
		grid = F.affine_grid(theta, x.size())
		# 使用网格构建采样器
		x = F.grid_sample(x, grid)
		return x

	def forward(self, x):
		# 对输入进行变换
		x = self.stn(x)
		x = F.relu(F.max_pool2d(self.conv1(x), 2))
		x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
		x = x.view(-1, 320)
		x = F.relu(self.fc1(x))
		x = F.dropout(x, training=self.training)
		x = self.fc2(x)
		return F.log_softmax(x, dim=1)

# -------实例化网络
model = Net().to(device)

# -------训练网络
optimizer = optim.SGD(model.parameters(), lr=0.01)
def train(epoch):
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output = model(data)
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()
		if batch_idx % 500 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.item()))
def test():
	with torch.no_grad():
		model.eval()
		test_loss = 0
		correct = 0
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			test_loss += F.nll_loss(output, target, size_average=False).item()
			pred = output.max(1, keepdim=True)[1]
			# 判断预测值与目标是否相等
			correct += pred.eq(target.view_as(pred)).sum().item()
		test_loss /= len(test_loader.dataset)
		print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
			.format(test_loss, correct, len(test_loader.dataset),
			100. * correct / len(test_loader.dataset)))

# -------显示运行
# tensor转numpy
def convert_image_np(inp):
	inp = inp.numpy().transpose((1, 2, 0))
	# 因为在加载数据时进行过标准归一化
	# 所以在这里需要反归一化
	mean = np.array([0.485, 0.456, 0.406])
	std = np.array([0.229, 0.224, 0.225])
	# 归一化：inp = (inp-mean)/std
	# 反归一化：inp = std * inp + mean
	inp = std * inp + mean
	inp = np.clip(inp, 0, 1)
	return inp

# 可视化
def visualize_stn():
	with torch.no_grad():
		# 取得测试集的第一个批次
		data = next(iter(test_loader))[0].to(device)
		# 把数据转回到cpu才能用pil显示
		input_tensor = data.cpu()
		# 此时model里边的参数以及被训练完成
		transformed_input_tensor = model.stn(data).cpu()
		in_grid = convert_image_np(torchvision.utils.make_grid(input_tensor))
		out_grid = convert_image_np(torchvision.utils.make_grid(transformed_input_tensor))
		# 并排绘制结果
		f, axarr = plt.subplots(1, 2)
		axarr[0].imshow(in_grid)
		axarr[0].set_title('Dataset Images')
		axarr[1].imshow(out_grid)
		axarr[1].set_title('Transformed Images')

# -------运行
for epoch in range(1, 6):
	train(epoch)
	test()

visualize_stn()
plt.show()