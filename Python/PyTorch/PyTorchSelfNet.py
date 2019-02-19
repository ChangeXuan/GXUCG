# -------从头构建一个MNIST识别网络，以便更好的认识Torch.nn
# -------下载MNIST数据集到本地
from pathlib import Path
import requests

DATA_PATH = Path("data")
# 表示为PATH=data\mnist
PATH = DATA_PATH / "mnist"
# 在代码文件根目录下创建data\mnist
PATH.mkdir(parents=True, exist_ok=True)
URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"
# 如果数据集不存在则进入if
if not (PATH/FILENAME).exists():
	print("Start Loading...")
	# 下载文件到内存
	content = requests.get(URL + FILENAME).content
	# 把内存中的文件写入本地文件
	(PATH/FILENAME).open("wb").write(content)

# -------读出数据
# pickle是用于序列化数据的一种特定于python的格式
import pickle
import gzip
# 这个数据集是采用numpy数组
# as_posize()将window样式改为UNIX样式
with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
	((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

# -------重新格式化数据
from matplotlib import pyplot
import numpy as np
# 数据集中，每张图片的大小为28*28，但是储存是用的是一维，即784(28*28)
# 所以需要将一维数据还原为2二维
pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
# pyplot.show()
# print(x_train.shape)

# -------把numpy的数据转换为tensor
import torch
# 使用map和torch.tensor批量把numpy数据转换为tensor
x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))
# n为图片数量，c为一维图片的长度
n, c = x_train.shape
# print(x_train, y_train)
# print(x_train.shape)
# print(y_train.min(), y_train.max())

# -------从头开始构建一个神经网络
import math
# 尺寸大小为784*10(784行，10列)
weights = torch.randn(784, 10) / math.sqrt(784)
# 设置需要自动求梯度
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)

# -------自己编写线性模型
def log_softmax(x):
	# log(exp(x)/sum(exp(x)))
	return x-x.exp().sum(-1).log().unsqueeze(-1)
def model(xb):
	# @表示点积操作
	return log_softmax(xb @ weights + bias)

bs = 64
# 取出前64个数据
xb = x_train[0: bs]
preds = model(xb)
print(preds[0], preds.shape)

# -------使用负对数似然来做损失函数
def nll(input, target):
	# mean()为返回平均值
	return -input[range(target.shape[0]), target].mean()
loss_func = nll

# -------检查随机权重的损失
yb = y_train[0: bs]
print(loss_func(preds, yb))

# -------计算精确度
def accuracy(out, yb):
	# 返回张量上最大值的下标
	preds = torch.argmax(out, dim=1)
	# 以次判断每一项，若相等为1，不等为0，最后求均值
	return (preds == yb).float().mean()

# -------检查随机权重的精确度
print(accuracy(preds, yb))

# -------对网络进行迭代训练
'''
- 选择小批次的大小（这里的bs为64，即每次训练64张)
- 使用模型去预测输出
- 计算损失
- 反向传播来更新权重这里是weights和bias
'''
# 这个是python单步调试器
from IPython.core.debugger import set_trace
lr = 0.5
epochs = 2

for epoch in range(epochs):
	print('epoch: %d / epochs:'%epoch)
	# 每次64张，共循环多少次能够用完50000张图片
	for i in range((n-1) // bs + 1):
		# set_trace()
		start_i = i*bs
		end_i = start_i+bs
		xb = x_train[start_i:end_i]
		yb = y_train[start_i:end_i]
		preds = model(xb)
		loss = loss_func(preds, yb)
		print('loss is %f'%loss)
		loss.backward()
		with torch.no_grad():
			weights -= weights.grad*lr
			bias -= bias.grad*lr
			weights.grad.zero_()
			bias.grad.zero_()
	print('-'*10)
# -------到此，我们搭建了一个没有隐藏层的只有一个softmax输出的最简单的分类网络
print(loss_func(model(xb), yb), accuracy(model(xb), yb))
