# -----------使用Adam的VAE
from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

# -------网络模型
class VAE(nn.Module):
	def __init__(self):
		super(VAE, self).__init__()
		# 28*28=784
		self.fc1 = nn.Linear(784, 400)
		self.fc21 = nn.Linear(400, 20)
		self.fc22 = nn.Linear(400, 20)
		self.fc3 = nn.Linear(20, 400)
		self.fc4 = nn.Linear(400, 784)
	# 编码
	def encode(self, x):
		h1 = F.relu(self.fc1(x))
		return self.fc21(h1), self.fc22(h1)
	# 重新参数化
	def reparameterize(self, mu, logvar):
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		return eps.mul(std).add_(mu)
	# 解码
	def decode(self, z):
		h3 = F.relu(self.fc3(z))
		return torch.sigmoid(self.fc4(h3))
	# 前向传播
	def forward(self, x):
		mu, logvar = self.encode(x.view(-1, 784))
		z = self.reparameterize(mu, logvar)
		return self.decode(z), mu, logvar


# -------获取数据
def load_data():
	train_loader = torch.utils.data.DataLoader(
					datasets.MNIST('./data/data', train=True, download=True,
									transform=transforms.ToTensor()),
					batch_size=128, shuffle=True)
	test_loader = torch.utils.data.DataLoader(
					datasets.MNIST('./data/data', train=True, download=True,
									transform=transforms.ToTensor()),
					batch_size=128, shuffle=True)
	return train_loader, test_loader

# -------损失函数
def loss_function(recon_x, x, mu, logvar):
	# 这一项为重建误差
	# recon_x为重建后的数据，x为输入的数据
	BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
	# 这一项为kl误差
	# 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
	KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	return BCE + KLD

# -------训练函数
def train(epoch, model, device, train_loader, optimizer):
	model.train()
	train_loss = 0
	for batch_idx, (data, _) in enumerate(train_loader):
		data = data.to(device)
		optimizer.zero_grad()
		recon_batch, mu, logvar = model(data)
		loss = loss_function(recon_batch, data, mu, logvar)
		loss.backward()
		train_loss += loss.item()
		optimizer.step()
		if batch_idx % 10 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader),
				loss.item() / len(data)))
	print('====> Epoch: {} Average loss: {:.4f}'.format( 
		epoch, train_loss / len(train_loader.dataset)))

# -------测试函数
def test(epoch, model, device, test_loader):
	model.eval()
	test_loss = 0
	with torch.no_grad():
		for i, (data, _) in enumerate(test_loader):
			data = data.to(device)
			recon_batch, mu, logvar = model(data)
			test_loss += loss_function(recon_batch, data, mu, logvar).item()
			if i == 0:
				pass
	test_loss /= len(test_loader.dataset)
	print('====> Test set loss: {:.4f}'.format(test_loss))

# -------主函数
if __name__ == "__main__":
	# 选择运行设备
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	# 实例模型
	model = VAE().to(device)
	# 设置优化器
	optimizer = optim.Adam(model.parameters(), lr=1e-3)
	# 加载训练和测试数据集迭代器
	train_loader, test_loader = load_data()
	for epoch in range(1, 11):
		train(epoch, model, device, train_loader, optimizer)
		test(epoch, model, device, test_loader)
		with torch.no_grad():
			sample = torch.randn(64, 20).to(device)
			de_sample = model.decode(sample).cpu()
			# 64张1通道28*28
			save_image(de_sample.view(64, 1, 28, 28), './results/sample_' + str(epoch) + '.png')