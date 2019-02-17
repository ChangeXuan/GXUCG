from __future__ import print_function
import torch
import numpy as np

# # 构造未初始化的5行3列的张量(默认全为0)
# x = torch.empty(5, 3)
# print(x)

# # 构造随机的5行3列的张量
# x = torch.rand(5, 3)
# print(x)

# # 构造全部是0的5行3列的张量，元素的数据类型为torch的long型
# x = torch.zeros(5, 3,dtype=torch.long)
# print(x)

# # 直接使用数据来构造一个张量
# x = torch.tensor([5.5, 3])
# print(x)
# # 使用已经存在的张量来初始化一个新的张量
# # new_ones表示初始化的值为1
# x = x.new_ones(5, 3, dtype=torch.double)
# print(x)
# x = torch.randn_like(x, dtype=torch.float)
# print(x)

# # 取得张量的大小
# # torch.Size是一个元组，所以支持所有元组的操作
# print(x.size())

# x = torch.ones(5, 3)
# # 随机生成元素为0-1的5行3列的张量
# y = torch.rand(5, 3)
# # 第一种加法操作
# print(x+y)

# 第二种加法操作
# print(torch.add(x, y))

# 使用参数输出来接收加法结果
# result = torch.empty(5, 3)
# torch.add(x, y, out=result)
# print(result)

# 第三种加法操作
# y.add_(x)
# print(y)

# 切片操作
# 第一个参数为行，这里是全取
# 第二个参数为列，这里是只取第2列
# print(y)
# print(y[:,2])

# # 改变张量的大小
# # 原来为4x4
# x = torch.randn(4, 4)
# # 调整为1x16
# y = x.view(16)
# # -1为推断，调整为2x8
# z = x.view(-1,8)
# print(x.size(), y.size(), z.size())

# # 如果张量中只有一个元素，着可以取得张量中的值
# x = torch.randn(1)
# print(x)
# print(x.item())

# # torch的张量转numpy的array
# a = torch.ones(5)
# b = a.numpy()
# # 因为torch与怒骂朋友共享内存，所以当a改变时，b也会发生变化
# a.add_(1)
# print(a)
# print(b)

# # numpy的array转torch的张量
# a = np.ones(5)
# b = torch.from_numpy(a)
# # 因为共享内存，所以你变我也变
# np.add(a, 1, out=a)
# print(a)
# print(b)

# x = torch.ones(5, 3)
# y = torch.rand(5, 3)
# # 判断cuda是否可用
# if torch.cuda.is_available():
# 	# 声明一个CUDA设备对象
# 	device = torch.device("cuda")
# 	# 在GPU上直接创建一个张量
# 	y = torch.ones_like(x, device=device)
# 	# 把x张量转移到GPU上
# 	x = x.to(device)
# 	z = x + y
# 	# 此时z在GPU上，
# 	print(z)
# 	# 把z移动到CPU上并设置类型
# 	print(z.to("cpu", torch.double))

# # 创建一个张量并设置跟踪计算梯度
# x = torch.ones(2, 2, requires_grad=True)
# print(x)
# y = x + 2
# print(y)
# # 因为y是运算出来的，所以y有grad_fn
# # 因为x是认为创建的，所以x的grad_fn为None
# print(y.grad_fn)
# z = y * y * 3
# out = z.mean()
# print(z, out)
# # 求梯度
# # 张量o = 1/(2*2)*sum(z)
# # z = 3*y^2
# # z = 3*(x+2)^2
# # 设置out为被导函数，x为导变量
# out.backward()
# print(x.grad)

# a = torch.randn(2, 2)
# a = ((a * 3) / (a - 1))
# print(a.requires_grad)
# # 需要设置后才为True
# a.requires_grad_(True)
# print(a.requires_grad)
# b = (a * a).sum()
# print(b.grad_fn)

# x = torch.randn(3, requires_grad=True)
# print(x)
# y = x * 2
# flag = 1
# # y.data.norm()是把y中的值平方求和后再开方
# while y.data.norm() < 1000:
# 	y = y * 2
# 	flag += 1
# print(y)
# print(flag)
# v = torch.tensor([0.2, 1.0, 0.0001], dtype=torch.float)
# # 相当于对该式子求x的偏导：y=2^n*x*v
# y.backward(v)
# print(x.grad)

# print(x.requires_grad)
# print((x ** 2).requires_grad)
# with torch.no_grad():
# 	print((x ** 2).requires_grad)

