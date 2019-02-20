# -------深入细化和提取特征
'''
一般来说，迁移学习主要考虑下面4步
- 初始化预训练模型
- 重新构造最后的输出层，保证与自己的数据集相同
- 定义在训练过程中需要更新哪些参数的优化算法
- 运行并训练
'''
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

# -------设置参数
data_dir = "./data/hymenoptera_data"
# 预训练模型，可以从中选择[resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "squeezenet"
# 数据集类别(这里有训练集和验证集)
num_classes = 2
batch_size = 8
num_epochs = 15
# 特征提取标志，为False时会对整个模型进行调整
# 为True时，只更新添加层的参数
feature_extract = True

# -------定义模型训练代码
# 最后一个参数时inceptionV3模型标准参数
def train_model(device, model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
	start_time = time.time()

	val_acc_history = []
	# 模型最好的参数
	best_model_wts = copy.deepcopy(model.state_dict())
	# 模型最好的准确率
	best_acc = 0.0

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-'*10)
		# 每个周期都会有训练阶段和验证阶段
		for phase in ['train', 'val']:
			if phase == 'train':
				model.train()
			else:
				model.eval()
			running_loss = 0.0
			running_corrects = 0

			# 迭代数据，每次数据数量为batch_size
			for inputs, labels in dataloaders[phase]:
				inputs, labels = inputs.to(device), labels.to(device) 
				optimizer.zero_grad()
				# 是训练阶段就开启自动梯度，是验证阶段则不开启
				with torch.set_grad_enabled(phase == 'train'):
					if is_inception and phase == 'train':
						# 第二个为辅助输出层
						outputs, aux_outputs = model(inputs)
						loss1 = criterion(outputs, labels)
						loss2 = criterion(aux_outputs, labels)
						loss = loss1 + 0.4*loss2
					# 当is_inception为False时
					# 当为验证阶段时，无论如何都会进入该else
					else:
						outputs = model(inputs)
						loss = criterion(outputs, labels)
					_, preds = torch.max(outputs, 1)
					if phase == 'train':
						loss.backward()
						optimizer.step()
				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(preds == labels.data)
			epoch_loss = running_loss / len(dataloaders[phase].dataset)
			epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
			print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
			if phase == 'val' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())
			if phase == 'val':
				val_acc_history.append(epoch_acc)

	end_time = time.time() - start_time
	print('Training complete in {:.0f}m {:.0f}s'.format(end_time // 60, end_time % 60))
	print('Best val Acc: {:4f}'.format(best_acc))
	# 加载最好的权重到模型
	model.load_state_dict(best_model_wts)
	return model,val_acc_history

# -------设置参数是否进行自动求梯度(是否冻结模型参数)
def set_parameter_requires_grad(model, feature_extract):
	# 如果时特征提取，则冻结所有参数
	if feature_extract:
		for param in model.parameters():
			param.requires_grad = False

# -------初始化预加载模型
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = 0
    # 其他网络要求输入为224，inception要求输入为299
    if model_name == "resnet":
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # 处理辅助网络
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # 处理主网络
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

# -------加载数据
def load_data(input_size, data_dir):
	# 数据的增强和规范化
	data_transforms = {
		'train': transforms.Compose([
			transforms.RandomResizedCrop(input_size),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			# 第一个是均值，第二个是标准差
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]),
		'val': transforms.Compose([
			transforms.Resize(input_size),
			transforms.CenterCrop(input_size),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]),
	}
	print("Initializing Datasets and Dataloaders...")
	# 创建数据集
	image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
	# 创建数据集迭代器
	dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'val']}
	return dataloaders_dict

# -------初始化模型
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=True)
# -------设置运行设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# -------加载数据
dataloaders_dict = load_data(input_size, data_dir)

model_ft = model_ft.to(device)
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
	params_to_update = []
	for name, param in model_ft.named_parameters():
		if param.requires_grad == True:
			params_to_update.append(param)
			print("\t", name)
else:
	for name, param in model_ft.named_parameters():
		if param.requires_grad == True:
			print("\t",name)
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
model_ft, hist = train_model(device, model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))

# -------不使用转移学习
# 即使用网络，但是不使用学习好的参数
# 即使用现有的数据集来学习参数
scratch_model, _ = initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=False)
scratch_model = scratch_model.to(device)
scratch_optimizer = optim.SGD(scratch_model.parameters(), lr=0.001, momentum=0.9)
scratch_criterion = nn.CrossEntropyLoss()
_, scratch_hist = train_model(device, scratch_model, dataloaders_dict, scratch_criterion, scratch_optimizer, num_epochs=num_epochs, is_inception=(model_name=="inception"))


ohist = []
shist = []
ohist = [h.cpu().numpy() for h in hist]
shist = [h.cpu().numpy() for h in scratch_hist]
plt.title("Validation Accuracy vs. Number of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Validation Accuracy")
plt.plot(range(1,num_epochs+1),ohist,label="Pretrained")
plt.plot(range(1,num_epochs+1),shist,label="Scratch")
plt.ylim((0,1.))
plt.xticks(np.arange(1, num_epochs+1, 1.0))
plt.legend()
plt.show()