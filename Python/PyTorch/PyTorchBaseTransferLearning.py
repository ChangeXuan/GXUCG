# -------使用训练好的网络完成学习的迁移
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

# -------数据变换
data_transforms = {
	'train': transforms.Compose([
		transforms.RandomResizedCrop(224),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		# 使用均值和标准差进行归一化
		# 公式为x = (x-mean) / std
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
	'val': transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
}
# -------加载数据
data_dir = 'data/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True) for x in ['train', 'val']}
# 读取底层文件的数目
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# 取得'train'文件夹内的文件夹的名字
class_names = image_datasets['train'].classes
# 如果GPU可以使用则使用，否则使用cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# -------可视化几张图片test
def imshow(inp, title=None):
	inp = inp.numpy().transpose((1, 2, 0))
	mean = np.array([0.485, 0.456, 0.406])
	std = np.array([0.229, 0.224, 0.225])
	# 反均一化
	inp = std*inp + mean
	# 剪切剩0到1
	# ???
	inp = np.clip(inp, 0, 1)
	plt.imshow(inp)
	plt.title(title)
	plt.show()
# # 1st
# # dataiter = iter(dataloaders['train'])
# # inputs, classes = dataiter.next()
# # 2st
# inputs, classes = next(iter(dataloaders['train']))
# out = torchvision.utils.make_grid(inputs)
# imshow(out, title=[class_names[x] for x in classes])
# -------构造一个一般训练模型
# scheduler是一个LR调度对象
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
	since = time.time()
	# model.state_dict()返回模型的状态，包括参数
	# 可以使用model.state_dict().keys()查看参数信息
	# copy.deepcopy进行深度拷贝
	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		# 输出10个'-'
		print('-' * 10)
		for phase in ['train', 'val']:
			if phase == 'train':
				# 更新lr学习速率
				scheduler.step()
				model.train()
			else:
				model.eval()
			running_loss = 0.0
			running_corrects = 0
			for inputs, labels  in dataloaders[phase]:
				# 把张量数据放入GPU中
				inputs = inputs.to(device)
				labels = labels.to(device)
				# 清空优化器的梯度
				optimizer.zero_grad()
				# 设置是否开启或禁用梯度
				# 当phase为'train'时是开启
				with torch.set_grad_enabled(phase=='train'):
					outputs = model(inputs)
					# 找到给定维数中最大值
					# 这里的dim为1，则找出每行中的最大值
					# out1是最大值，out2是最大值所在的下标index
					_, preds = torch.max(outputs, 1)
					loss = criterion(outputs, labels)
					# 如果是训练阶段则反向传播并更新参数
					if phase == 'train':
						loss.backward()
						optimizer.step()
				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(preds == labels.data)
			epoch_loss = running_loss / dataset_sizes[phase]
			epoch_acc = running_corrects.double() / dataset_sizes[phase]
			print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
			if phase == 'val' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))
	model.load_state_dict(best_model_wts)
	return model

# -------可视化模型
def visualize_model(model, num_images=6):
	was_training = model.training
	# 将模型设置为评估模式
	model.eval()
	images_so_far = 0
	fig = plt.figure()
	with torch.no_grad():
		for i, (inputs, labels) in enumerate(dataloaders['val']):
			inputs = inputs.to(device)
			labels = inputs.to(device)
			outputs = model(inputs)
			_, preds = torch.max(outputs, 1)
			for j in range(inputs.size()[0]):
				images_so_far += 1
				ax = plt.subplot(num_images//2, 2, images_so_far)
				ax.axis('off')
				print('predicted: {}'.format(class_names[preds[j]]))
				imshow(inputs.cpu().data[j])
				if images_so_far == num_images:
					# ???
					model.train(mode=was_training)
					return
		model.train(mode=was_training)

# # -------加载预先训练好的模型并重置全连接层
# # 取得一个在ImageNet上预先训练好的resnet18网络
# model_ft = models.resnet18(pretrained=True)
# # 构建新的最后一层
# num_ftrs = model_ft.fc.in_features
# # 设置最后一层
# model_ft.fc = nn.Linear(num_ftrs, 2)
# model_ft = model_ft.to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
# # 设置LR为每7个周期衰减0.1倍
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
# # 开始训练
# model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=3)
# visualize_model(model_ft)

# -------加载预先训练好的模型并冻结其他层，只训练最后一层
# 取得一个在ImageNet上预先训练好的resnet18网络
model_conv = models.resnet18(pretrained=True)
# 冻结原有模型上的全部参数
for param in model_conv.parameters():
	# 关闭自动求导即是冻结
	param.requires_grad = False
# 构建新的模块，默认情况下，构建的新模块的requires_grad=True
num_ftrs = model_conv.fc.in_features
# 设置最后一层
model_conv.fc = nn.Linear(num_ftrs, 2)
model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_conv.parameters(), lr=0.001, momentum=0.9)
# 设置LR为每7个周期衰减0.1倍
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
# 开始训练
model_conv = train_model(model_conv, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=3)
visualize_model(model_conv)