# -------保存和加载模型
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# -------自定义模型
class TheModelClass(nn.Module):
	def __init__(self):
		super(TheModelClass, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 5 * 5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

model = TheModelClass()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
print("Model's state_dict:")
# 输出模型的参数字典
for param_tensor in model.state_dict():
	print(param_tensor, "\t", model.state_dict()[param_tensor].size())
print("Optimizer's state_dict:")
# 输出优化器的参数字典
for var_name in optimizer.state_dict():
	print(var_name, "\t", optimizer.state_dict()[var_name])

# # -------保存和加载参数模型
# # 把模型序列化到文件
# torch.save(model.state_dict(),'./model.pt')
# model = TheModelClass()
# # 先反序列化到内存然后加载到model
# model.load_state_dict(torch.load('./model.pt'))
# # 在预测前需要把model设置为评估模式
# model.eval()

# # -------保存和加载全部模型
# local_model = None
# torch.save(model, './model1.pt')
# local_model = torch.load('./model1.pt')
# local_model.eval()
# print(local_model)
# for param_tensor in local_model.state_dict():
# 	print(param_tensor, "\t", local_model.state_dict()[param_tensor].size())

# # -------保存和加载检查点
# torch.save({
# 			'epoch': epoch,
# 			'model_state_dict': model.state_dict(),
# 			'optimizer_state_dict': optimizer.state_dict(),
# 			'loss': loss,
# 			...
# 			}, PATH)# 文件类型推荐使用.tar

# model = TheModelClass()
# optimizer = TheOptimizerClass()
# checkpoint = torch.load(PATH)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']

# model.eval()
# # 如果你还想训练
# model.train()

# # -------保存多个模型到一个文件和加载
# torch.save({
# 			'modelA_state_dict': modelA.state_dict(),
# 			'modelB_state_dict': modelB.state_dict(),
# 			'optimizerA_state_dict': optimizerA.state_dict(),
# 			'optimizerB_state_dict': optimizerB.state_dict(),
# 			...
# 			}, PATH)# 推荐格式为.tar
# modelA = TheModelAClass(*args, **kwargs)
# modelB = TheModelBClass(*args, **kwargs)
# optimizerA = TheOptimizerAClass(*args, **kwargs)
# optimizerB = TheOptimizerBClass(*args, **kwargs)
# checkpoint = torch.load(PATH)
# modelA.load_state_dict(checkpoint['modelA_state_dict'])
# modelB.load_state_dict(checkpoint['modelB_state_dict'])
# optimizerA.load_state_dict(checkpoint['optimizerA_state_dict'])
# optimizerB.load_state_dict(checkpoint['optimizerB_state_dict'])
# modelA.eval()
# modelB.eval()
# modelA.train()
# modelB.train()

# # -------使用不同模型的参数
# torch.save(modelA.state_dict(), PATH)
# modelB = TheModelBClass()
# modelB.load_state_dict(torch.load(PATH), strict=False)