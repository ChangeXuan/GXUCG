# # -------使用numpy搭建网络(手动求导)test
# import numpy as np
# # N是批次大小，D_in是输入的维度，H是隐藏层的维度，D_out是输出维度
# N, D_in, H, D_out = 64, 1000, 100, 10
# # 创建随机输入和输出
# x = np.random.randn(N, D_in)
# y = np.random.randn(N, D_out)
# # 随机初始化权重
# w1 = np.random.randn(D_in, H)
# w2 = np.random.randn(H, D_out)
# # 设置学习率
# learning_rate = 1e-6
# for t in range(500):
# 	# ---前向传播
# 	h = x.dot(w1)
# 	h_relu = np.maximum(h, 0)
# 	y_pred = h_relu.dot(w2)
# 	# 计算loss
# 	loss = np.square(y_pred - y).sum()
# 	print(t,loss)
# 	# ---反向传播
# 	grad_y_pred = 2.0*(y_pred-y)
# 	grad_w2 = h_relu.T.dot(grad_y_pred)
# 	grad_h_relu = grad_y_pred.dot(w2.T)
# 	grad_h = grad_h_relu.copy()
# 	grad_h[h<0]=0
# 	grad_w1 = x.T.dot(grad_h)
# 	# 更新权重
# 	w1 -= learning_rate*grad_w1
# 	w2 -= learning_rate*grad_w2


# # -------使用torch来搭建网络(手动求导)test
# import torch
# dtype = torch.float
# # 使用CPU
# device = torch.device("cpu")
# # 使用GPU
# # device = torch.device("cuda:0")
# N, D_in, H, D_out = 64, 1000, 100, 10
# x = torch.randn(N, D_in, device=device, dtype=dtype)
# y = torch.randn(N, D_out, device=device, dtype=dtype)
# w1 = torch.randn(D_in, H, device=device, dtype=dtype)
# w2 = torch.randn(H, D_out, device=device, dtype=dtype)
# learning_rate=1e-6
# for t in range(500):
# 	h = x.mm(w1)
# 	h_relu = h.clamp(min=0)
# 	y_pred = h_relu.mm(w2)
# 	loss = (y_pred-y).pow(2).sum().item()
# 	print(t, loss)
# 	grad_y_pred = 2.0 * (y_pred - y)
# 	grad_w2 = h_relu.t().mm(grad_y_pred)
# 	grad_h_relu = grad_y_pred.mm(w2.t())
# 	grad_h = grad_h_relu.clone()
# 	grad_h[h < 0] = 0
# 	grad_w1 = x.t().mm(grad_h)
# 	w1 -= learning_rate*grad_w1
# 	w2 -= learning_rate*grad_w2


# # -------使用torch来搭建网络(自动求导)test
# import torch
# dtype = torch.float
# device = torch.device("cpu")
# N, D_in, H, D_out = 64, 1000, 100, 10
# # 这里的requires_grad默认为False,表示我们不需要对x和y求导
# x = torch.randn(N, D_in, device=device, dtype=dtype)
# y = torch.randn(N, D_out, device=device, dtype=dtype)
# # 这里的requires_grad设置为True，表示我们需要对w1和w2进行求导
# w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
# w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)
# learning_rate=1e-6
# for t in range(500):
# 	# 因为我们使用自动求导，所以中间值我们可以不保持引用
# 	y_pred = x.mm(w1).clamp(min=0).mm(w2)
# 	loss = (y_pred - y).pow(2).sum()
# 	print(t, loss.item())
# 	# 使用方向传播自动求导，因为我们标记了w1和w2，所以torch会帮我们求出
# 	loss.backward()
# 	# 禁用梯度计算，即关闭.backward()
# 	with torch.no_grad():
# 		w1 -= learning_rate * w1.grad
# 		w2 -= learning_rate * w2.grad
# 		# 梯度清零
# 		w1.grad.zero_()
# 		w2.grad.zero_()


# # -------使用自定义求导函数来搭建网络test
# import torch
# class MyReLU(torch.autograd.Function):
# 	@staticmethod
# 	def forward(ctx, input):
# 		ctx.save_for_backward(input)
# 		return input.clamp(min=0)
# 	@staticmethod
# 	def backward(ctx, grad_output):
# 		input, _ = ctx.saved_tensors
# 		grad_input = grad_output.clone()
# 		grad_input[input<0]=0
# 		return grad_input

# dtype = torch.float
# device = torch.device("cpu")
# N, D_in, H, D_out = 64, 1000, 100, 10
# x = torch.randn(N, D_in, device=device, dtype=dtype)
# y = torch.randn(N, D_out, device=device, dtype=dtype)
# w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
# w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)
# learning_rate = 1e-6
# for t in range(500):
# 	relu = MyReLU.apply
# 	# 这里会调用我们定义在MyReLU中的forward函数
# 	y_pred = relu(x.mm(w1)).mm(w2)
# 	loss = (y_pred - y).pow(2).sum()
# 	print(t, loss.item())
# 	# 这里会调用我们定义在MyReLU中的backward函数
# 	loss.backward()
# 	with torch.no_grad():
# 		w1 -= learning_rate * w1.grad
# 		w2 -= learning_rate * w2.grad
# 		w1.grad.zero_()
# 		w2.grad.zero_()


# # -------使用nn包搭建网络test
# import torch
# N, D_in, H, D_out = 64, 1000, 100, 10
# x = torch.randn(N, D_in)
# y = torch.randn(N, D_out)
# # 这里的Linear的公式为y=wx+b，故参数为w和b
# model = torch.nn.Sequential(
# 	torch.nn.Linear(D_in, H),
# 	torch.nn.ReLU(),
# 	torch.nn.Linear(H, D_out),
# 	)
# loss_fn = torch.nn.MSELoss(reduction='sum')
# learning_rate = 1e-4
# for t in range(500):
# 	y_pred = model(x)
# 	loss = loss_fn(y_pred, y)
# 	print(t, loss.item())
# 	# 把模型中参数的梯度清零
# 	model.zero_grad()
# 	loss.backward()
# 	with torch.no_grad():
# 		# 遍历参数，更新参数
# 		for param in model.parameters():
# 			param -= learning_rate * param.grad


# # -------使用Adam优化算法test
# import torch
# N, D_in, H, D_out = 64, 1000, 100, 10
# x = torch.randn(N, D_in)
# y = torch.randn(N, D_out)
# model = torch.nn.Sequential(
#     torch.nn.Linear(D_in, H),
#     torch.nn.ReLU(),
#     torch.nn.Linear(H, D_out),
# )
# loss_fn = torch.nn.MSELoss(reduction='sum')

# learning_rate = 1e-4
# # 第一个参数告诉Adam应该优化那些参数
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# for t in range(500):
# 	y_pred = model(x)
# 	loss = loss_fn(y_pred, y)
# 	print(t, loss.item())
# 	optimizer.zero_grad()
# 	loss.backward()
# 	# 使用优化器调用step就会更新参数
# 	optimizer.step()


# # -------自定义nn模型
# import torch
# class TwoLayerNet(torch.nn.Module):
# 	def __init__(self, D_in, H, D_out):
# 		super(TwoLayerNet, self).__init__()
# 		self.linear1 = torch.nn.Linear(D_in, H)
# 		self.linear2 = torch.nn.Linear(H, D_out)

# 	def forward(self, x):
# 		h_relu = self.linear1(x).clamp(min=0)
# 		y_pred = self.linear2(h_relu)
# 		return y_pred

# N, D_in, H, D_out = 64, 1000, 100, 10
# x = torch.randn(N, D_in)
# y = torch.randn(N, D_out)
# model = TwoLayerNet(D_in, H, D_out)
# criterion = torch.nn.MSELoss(reduction='sum')
# # 使用SGD(随机梯度下降)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
# for t in range(500):
# 	y_pred = model(x)
# 	loss = criterion(y_pred, y)
# 	print(t, loss.item())
# 	optimizer.zero_grad()
# 	loss.backward()
# 	optimizer.step()


# # -------实现一个动态网络
# import random
# import torch
# class DynamicNet(torch.nn.Module):
# 	def __init__(self, D_in, H, D_out):
# 		super(DynamicNet, self).__init__()
# 		self.input_linear = torch.nn.Linear(D_in, H)
# 		self.middle_linear = torch.nn.Linear(H, H)
# 		self.output_linear = torch.nn.Linear(H, D_out)

# 	def forward(self, x):
# 		h_relu = self.input_linear(x).clamp(min=0)
# 		# 因为是动态网络，所以每次都会访问
# 		for _ in range(random.randint(0, 3)):
# 			h_relu = self.middle_linear(h_relu).clamp(min=0)
# 		y_pred = self.output_linear(h_relu)
# 		return y_pred

# N, D_in, H, D_out = 64, 1000, 100, 10
# x = torch.randn(N, D_in)
# y = torch.randn(N, D_out)
# model = DynamicNet(D_in, H, D_out)
# criterion = torch.nn.MSELoss(reduction='sum')
# # 因为这个奇怪的网络很难训练，所以我们选用带动量的SGD
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
# for t in range(500):
# 	y_pred = model(x)
# 	loss = criterion(y_pred, y)
# 	print(t, loss.item())
# 	optimizer.zero_grad()
# 	loss.backward()
# 	optimizer.step()