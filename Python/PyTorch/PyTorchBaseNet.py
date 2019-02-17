import torch
import torch.nn as nn
import torch.nn.functional as F
'''
主要流程如下
input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
	  -> view -> linear -> relu -> linear -> relu -> linear
	  -> MSELoss
	  -> loss
'''
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		# 定义两个卷积核
		# 表示：输入是1通道，输出是6通道，卷积核大小为5*5
		self.conv1 = nn.Conv2d(1, 6, 5)
		# 表示：输入是6通道，输出是16通道，卷积核大小为5*5
		self.conv2 = nn.Conv2d(6, 16, 5)
		# 定义仿射算子：y = Wx + b
		# 默认bias偏置为True,表示学习偏置
		# 对输入数据做线性变换，W,b为可学习的权值
		self.fc1 = nn.Linear(16*5*5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		# 第一层操作：对输入进行卷积->进行relu激活->使用2*2进行池化(向下采样)=x-(k-1)
		# 32*32->28*28(5*5卷积)
		# 28*28->14*14(2*2池化)
		x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
		# 第二层操作：对输入进行卷积->进行relu激活->使用2*2进行池化(向下采样)=x-(k-1)
		# 14*14->10*10(5*5卷积)
		# 10*10->5*5(2*2池化)
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
		# 把5*5*16展平成一维，即400
		x = x.view(-1, self.num_flag_features(x))
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

	# 把数据展平成一维大小
	def num_flag_features(self, x):
		size = x.size()[1:]
		num_feautres = 1
		for s in size:
			num_feautres *= s
		return num_feautres

net = Net()
print(net)

input = torch.randn(1, 1, 32, 32)
output = net(input)
target = torch.randn(10)
target = target.view(1, -1)
# 实例化loss函数
criterion = nn.MSELoss()
# 判定output和target之间的均方误差
loss = criterion(output, target)
print(loss)
# 将所有参数缓冲区的梯度归零
net.zero_grad()
print(net.conv1.bias.grad)
loss.backward()
print(net.conv1.bias.grad)

# # 参数更新公式为：W = W - learning_rate * gradient
# learning_rate = 0.01
# # 取得网络的参数
# for f in net.parameters():
# 	f.data.sub_(f.grad.data * learning_rate)

# 使用torch提供的optim梯度下降
import torch.optim as optim

# 创建一个优化器
optimizer = optim.SGD(net.parameters(), lr = 0.01)
optimizer.zero_grad()
for i in range(6):
	output = net(input)
	loss = criterion(output, target)
	loss.backward()
	# 更新参数
	optimizer.step()
	print(net.conv1.bias.grad)

# # 输入是一个32*32大小的图像
# # nSamples*nChannels*Height*Width
# input = torch.randn(1, 1, 32, 32)
# # 输出是一个大小为10的向量
# out = net(input)
# print(out)
# net.zero_grad()
# out.backward(torhc.randn(1, 10))

# # 使用均方差样例
# input = torch.randn(1, 1, 32, 32)
# output = net(input)
# target = torch.randn(10)
# print(target)
# target = target.view(1, -1)
# # 实例化loss函数
# criterion = nn.MSELoss()
# # 判定output和target之间的均方误差
# loss = criterion(output, target)
# print(loss)

# # 后退查看
# print(loss.grad_fn)  # MSELoss
# print(loss.grad_fn.next_functions[0][0])  # Linear
# print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

# # 参数样例
# params = list(net.parameters())
# print(len(params))
# for p in params:
# 	print(p.size())
# print(params[3].size())