# -----------DCGAN
# 使用下一版本的新特性
from __future__ import print_function
# 命令行解析
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


# -------初始化随机数据
# 设置随机数种子,设为定值确定了每次生成的结果一致
manualSeed = 999
# # 如果需要生成新的不一致的结果，可以使用
# manualSeed = random.randint(1, 10000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
# 设置torch的随机数种子
torch.manual_seed(manualSeed)


# -------定义一些输入信息
# 数据集的根目录
dataroot = "./data/celeba"
# 提取数据集的线程数
workers = 0
# 小批次的大小
batch_size = 128
# 训练图片的大小，即所有图片将被调整为这么大
image_size = 64
# 训练图片的通道数
nc = 3
# 潜在空间z的大小，即生成器的输入大小
nz = 100
# 生成器的特征映射的大小
ngf = 64
# 判别器的特征映射的大小
ndf = 64
# 训练次数
num_epochs = 5
# 学习速率
lr = 0.0002
# Adam梯度下降的超参数
beta1 = 0.5
# GPU的个数
ngpu = 1


# -------加载数据并测试可视化
# ImageFolder要求在根目录下需要有个子目录
# 创建数据集,并变化和增强数据
dataset = dset.ImageFolder(root=dataroot,
							transform=transforms.Compose([
								transforms.Resize(image_size),
								transforms.CenterCrop(image_size),
								transforms.ToTensor(),
								transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
							]))
# 创建数据加载器
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

# 选择设备
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# # 显示一批测试数据集
# real_batch = next(iter(dataloader))
# plt.figure(figsize=(8,8))
# plt.axis("off")
# plt.title("Training Images")
# # (1, 2, 0)为修正tensor与numpy之间的不同
# plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
# plt.show()


# -------权重初始化
# 自定义权重初始化，G和D都会使用到
def weights_init(m):
	# 取得类名
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		# 从正态分布中取值来填充weight.data
		# 其中正态分布的mean为0.0，std为0.02
		nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		nn.init.normal_(m.weight.data, 1.0, 0.02)
		# 使用0来填充bias.data
		nn.init.constant_(m.bias.data, 0)


# -------定义生成器模型
# 生成器G用于将潜在空间向量(z)映射到数据空间
'''
- 使用卷积转置层将z变换到输出
- 使用tanh函数限制范围到[-1, 1]
- 每个卷积转置层后都会有一个batch norm function(批标准化)
'''
class Generator(nn.Module):
	def __init__(self, ngpu):
		super(Generator, self).__init__()
		self.ngpu = ngpu
		self.main = nn.Sequential(
			# inC, outC, kernelS, stride(步幅), padding
			# 反卷积与卷积作用相反
			nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
			# num_feature
			nn.BatchNorm2d(ngf*8),
			nn.ReLU(True),
			# 需要走完全部数据，不足的地方用1填充
			nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
			nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf*2),
			nn.ReLU(True),
			nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf),
			nn.ReLU(True),
			nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
			nn.Tanh()
		)
	def forward(self, input):
		return self.main(input)

netG = Generator(ngpu).to(device)
# 多gpu
if (device.type == 'cuda') and (ngpu > 1):
	netG = nn.DataParallel(netG, list(range(ngpu)))
# 添加权重
# mean=0, stdev=0.2.
netG.apply(weights_init)
print(netG)


# -------定义判别器模型
'''
- 是一个二分类网络，图片为输入
- 用Sigmoid输出概率
- 使用跨区卷积而不是池化来下采样是一个很好的实践，因为它让网络学习自己的池化函数
- 批标准化和relu函数促进了健康的梯度流
'''
class Discriminator(nn.Module):
	def __init__(self, ngpu):
		super(Discriminator, self).__init__()
		self.ngpu = ngpu
		self.main = nn.Sequential(
			nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 4),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 8),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
			nn.Sigmoid()
		)
	def forward(self, input):
		return self.main(input)

netD = Discriminator(ngpu).to(device)
if (device.type == 'cuda') and (ngpu > 1):
	netD = nn.DataParallel(netD, list(range(ngpu)))
# 添加权重
# mean=0, stdev=0.2.
netD.apply(weights_init)
print(netD)


# -------定义损失函数和优化器
# l = -[y*log(x) + (1-y)*log(1-x)]
criterion = nn.BCELoss()
# 创建一个随机的固定噪声
# 64行100列,即64张图片
fixed_noise = torch.randn(64, nz, 1, 1, device=device)
# 设置真伪标签
real_label = 1
fake_label = 0
# 分别对生成器和判别器定义Adam优化器
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


# -------训练判别器和生成器(训练GAN的过程中不同的超参数很容易发生崩溃)
# 训练判别器
'''
先用训练集的数据训练log(D(x))
再用生成器的数据训练log(1-D(G(z)))
'''
# 数据如下
'''
- Loss_D:(log(D(x)) + log(D(G(z))))
- Loss_G:(log(D(G(z))))
- D(x):所有真实值批次的平均输出，从1衰减到0，0.5为最佳
- D(G(z)):所有虚假值批次的平均输出，从0增长到1，0.5为最佳 
'''
# 这些数组用来追踪训练过程
img_list = []
G_losses = []
D_losses = []
iters = 0
print("Starting Training Loop...")
for epoch in range(num_epochs):
	# 第二个参数为下标起始位置，默认为0
	# data包含两个数据，一个是input,一个是lable
	for i, data in enumerate(dataloader, 0):
		#######
		# 第一部分，更新D网络：最大化log(D(x)) + log(1 - D(G(z)))
		#######
		netD.zero_grad()
		# ---训练真实批次
		real_cpu = data[0].to(device)
		# 批次大小
		b_size = real_cpu.size(0)
		# 填充，把real_label的值填充到(b_size,)张量中，即把label设置为全1
		# (b_size,)表示只有一行
		label = torch.full((b_size,), real_label, device=device)
		# .view(-1)集中变成一行
		output = netD(real_cpu).view(-1)
		# 计算loss
		# 因为这里的图片都是真实的，所以需要把output往label(1)上靠
		errD_real = criterion(output, label)
		# 通过反向传播计算梯度
		errD_real.backward()
		D_x = output.mean().item()
		# ---训练假批次
		# 随机生成一批隐藏向量
		noise = torch.randn(b_size, nz, 1, 1, device=device)
		# 使用生成器生成假图片
		fake = netG(noise)
		# 把label设置为全0
		label.fill_(fake_label)
		# 用D对所有假批次进行分类
		# 其中fake不需要求梯度
		output = netD(fake.detach()).view(-1)
		# 因为这里的图片都是假的，所以需要把output往label(0)上靠
		errD_fake = criterion(output, label)
		errD_fake.backward()
		# 求均值
		D_G_z1 = output.mean().item()
		errD = errD_real + errD_fake
		# 更新D的优化器
		optimizerD.step()
		#######
		# 第二部分，更新G网络：最大化log(D(G(z)))
		#######
		netG.zero_grad()
		label.fill_(real_label)
		# 之前通过optimizerD更新了netD
		output = netD(fake).view(-1)
		# 我们希望生成的fake为真
		errG = criterion(output, label)
		errG.backward()
		D_G_z2 = output.mean().item()
		# 更新G的优化器
		optimizerG.step()

		# 输出训练状态
		if i % 50 == 0:
			print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
				% (epoch, num_epochs, i, len(dataloader),
					errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
		# 保存loss值方便只有的绘制
		G_losses.append(errG.item())
		D_losses.append(errD.item())
		# 每500批次储存一组生成的图片
		# 当训练结束时生成一组图片
		if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
			with torch.no_grad():
				fake = netG(fixed_noise).detach().cpu()
			img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
		iters += 1

# -------查看生成器和判别器的loss值
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


# # -------G生成过程的可视化
# fig = plt.figure(figsize=(8,8))
# plt.axis("off")
# ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
# ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
# HTML(ani.to_jshtml())


# -------查看真实图片和虚假图片
real_batch = next(iter(dataloader))
# 绘制真实图片
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))
# 绘制假图片，最后一组
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()