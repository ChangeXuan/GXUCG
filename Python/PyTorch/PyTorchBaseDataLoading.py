'''
数据的加载和处理
'''
from __future__ import print_function, division
import os
import torch
# 用于分析csv文件
import pandas as pd
# 对图像进行操作
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
# 忽略警告
import warnings
warnings.filterwarnings("ignore")
# plt.ion()

# # -------读取CSV标记数据test
# landmarks_frame = pd.read_csv('data/faces/face_landmarks.csv')
# # 第65个数据，从0开始计数
# n = 65
# # iloc通过行号来取行数据
# img_name = landmarks_frame.iloc[n, 0]
# # 取得所有一系列坐标信息
# landmarks = landmarks_frame.iloc[n, 1:].as_matrix()
# # 把坐标信息转换为x,y的坐标信息
# landmarks = landmarks.astype('float').reshape(-1, 2)
# print('Image name: {}'.format(img_name))
# print('Landmarks shape: {}'.format(landmarks.shape))
# print('First 4 Landmarks:\n {}'.format(landmarks[:4]))

# -------显示图片和标记test
def show_landmarks(image, landmarks):
	plt.imshow(image)
	# 绘制点第一第二个参数为x和y
	plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
	# 暂停以便更新图表
	plt.pause(0.001)

# plt.figure()
# # 使用名字加载图片
# show_landmarks(io.imread(os.path.join('data/faces/', img_name)), landmarks)
# plt.show()

# -------定义自定义数据集的类
class FaceLandmarksDataset(Dataset):
	# csv文件路径，图片路径
	def __init__(self, csv_file, root_dir, transform=None):
		self.landmarks_frame = pd.read_csv(csv_file)
		self.root_dir = root_dir
		self.transform = transform

	# 继承Dataset需要重写的函数，目的是返回数据集的大小
	def __len__(self):
		return len(self.landmarks_frame)

	# 继承Dataset需要重写的函数，目的是能够按下标取值
	def __getitem__(self, idx):
		img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
		image = io.imread(img_name)
		landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
		landmarks = landmarks.astype('float').reshape(-1, 2)
		sample = {'image': image, 'landmarks': landmarks}
		if self.transform:
			sample = self.transform(sample)
		return sample

# -------使用自定义的数据集类test
# face_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
# 									root_dir='data/faces/')
# fig = plt.figure()
# for i in range(len(face_dataset)):
# 	sample = face_dataset[i]
# 	print(i, sample['image'].shape, sample['landmarks'].shape)
# 	# 绘制1*4的图片区域，坐标为第一行第i+1列
# 	ax = plt.subplot(1, 4, i+1)
# 	# 自动调整子图参数，使之填充整个图像区域
# 	plt.tight_layout()
# 	# 把i格式化到{}中
# 	ax.set_title('Sample #{}'.format(i))
# 	ax.axis('off')
# 	# 因为sample是个有两个键的字典，使用**传入函数，python会自动解析为两个参数
# 	show_landmarks(**sample)
# 	if i==3:
# 		plt.show()
# 		break

# -------缩放图片的类
class Rescale(object):
	def __init__(self, output_size):
		# 判断某个对象是否属于某个类
		# 判断output_size是否属于int或tuple
		assert isinstance(output_size, (int, tuple))
		self.output_size = output_size

	# __call__为类的可直接调用对象
	# R = Rescale()
	# R(xx) # 这句就会调用类中的__call__函数
	def __call__(self, sample):
		image, landmarks = sample['image'], sample['landmarks']
		# h,w,c
		h, w = image.shape[:2]
		# 如果输入的是int
		# 保持图片的长宽比，短边设置为输入尺寸
		if isinstance(self.output_size, int):
			if h > w:
				new_h, new_w = self.output_size * h / w, self.output_size
			else:
				new_h, new_w = self.output_size, self.output_size * w / h
		else:
			# 若是元组，则直接赋值
			new_h, new_w = self.output_size
		new_h, new_w = int(new_h), int(new_w)
		img = transform.resize(image, (new_h, new_w))
		# 标记坐标减少相应的比例
		landmarks = landmarks * [new_w / w, new_h / h]
		return {'image': img, 'landmarks': landmarks}

# -------随机对图像进行剪裁(这是数据增强)
class RandomCrop(object):
	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		# 如果输入是整数，则剪切尺寸为方形
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size
	def __call__(self, sample):
		image, landmarks = sample['image'], sample['landmarks']
		h, w = image.shape[:2]
		new_h, new_w = self.output_size
		top = np.random.randint(0, h - new_h)
		left = np.random.randint(0, w - new_w)
		image = image[top:top+new_h, left:left+new_w]
		landmarks = landmarks - [left, top]
		return {'image': image, 'landmarks': landmarks}
		

# -------将numpy图像转为torch图像(需要翻转轴)
class ToTensor(object):
	def __call__(self, sample):
		image, landmarks = sample['image'], sample['landmarks']
		# numpy的图片和torch的图片的轴不同
		# numpy image: H*W*C
		# torch image: C*H*W
		image = image.transpose((2, 0, 1))
		return {'image': torch.from_numpy(image), 'landmarks': torch.from_numpy(landmarks)}


# -------对一个样本进行变换test
# scale = Rescale(256)
# crop = RandomCrop(128)
# composed = transforms.Compose([Rescale(256), RandomCrop(224)])

# fig = plt.figure()
# sample = face_dataset[65]
# for i, tsfrm in enumerate([scale, crop, composed]):
# 	transformed_sample = tsfrm(sample)
# 	ax = plt.subplot(1, 3, i+1)
# 	plt.tight_layout()
# 	ax.set_title(type(tsfrm).__name__)
# 	show_landmarks(**transformed_sample)
# plt.show()


# # -------取得已经转换好的样本
# # 对每个样本先缩放到256*256，再剪裁到224*224
# transformed_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
# 											root_dir='data/faces/',
# 											transform=transforms.Compose([
# 												Rescale(256),
# 												RandomCrop(224),
# 												ToTensor()
# 											]))
# for i in range(len(transformed_dataset)):
# 	sample = transformed_dataset[i]
# 	print(i, sample['image'].size(), sample['landmarks'].size())
# 	if i==3:
# 		break

# -------使用DataLoader来加载数据
transformed_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
											root_dir='data/faces/',
											transform=transforms.Compose([
												Rescale(256),
												RandomCrop(224),
												ToTensor()
											]))
dataloader = DataLoader(transformed_dataset, batch_size=4,
						shuffle=True, num_workers=0)
# 帮助显示batch
def show_landmarks_batch(sample_batched):
	images_batch, landmarks_batch = sample_batched['image'], sample_batched['landmarks']
	batch_size = len(images_batch)
	# 这里的images_batch为:4*3*224*224
	# 4张图片，3通道
	im_size = images_batch.size(2)
	grid = utils.make_grid(images_batch)
	plt.imshow(grid.numpy().transpose(1, 2, 0))
	for i in range(batch_size):
		plt.scatter(landmarks_batch[i, :, 0].numpy() + i*im_size,
					landmarks_batch[i, :, 1].numpy(),
					s=10, marker='.', c='r')
		plt.title('Batch from dataloader')

for i_batch, sample_batched in enumerate(dataloader):
	print(i_batch, sample_batched['image'].size(),
			sample_batched['landmarks'].size())
	if i_batch == 3:
		plt.figure()
		show_landmarks_batch(sample_batched)
		plt.axis('off')
		plt.ioff()
		plt.show()
		break


# # -------使用系统提供的转换类
# '''
# root/ants/xxx.png
# root/ants/xxy.jpeg
# root/ants/xxz.png
# .
# .
# root/bees/123.jpg
# root/bees/nsdf3.png
# root/bees/asd932_.png
# '''
# import torch
# from torchvision import transforms, datasets
# data_transform = transforms.Compose([
#         transforms.RandomSizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])
# # 
# hymenoptera_dataset = datasets.ImageFolder(root='hymenoptera_data/train',
# 											transform=data_transform)
# dataset_loader = torch.utils.data.DataLoader(hymenoptera_dataset,
# 											batch_size=4, shuffle=True,
# 											num_workers=0)