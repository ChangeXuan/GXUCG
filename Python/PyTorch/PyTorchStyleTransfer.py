# ------------使用神经网络进行图片的风格迁移
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import copy

# -------设置运行设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------加载和预处理数据
# 如果GPU无法使用，则使用较小尺寸
imsize = 512 if torch.cuda.is_available() else 128
# torch.tensor只支持[0,1]
loader = transforms.Compose([
	transforms.Resize(imsize),
	transforms.ToTensor()])
# 加载图片函数
def image_loader(image_name):
	image = Image.open(image_name)
	# unsqueeze(0)把数据压缩成一行？
	image = loader(image).unsqueeze(0)
	return image.to(device, torch.float)
# 读取图片
style_img = image_loader("./data/images/neural-style/picasso.jpg")
content_img = image_loader("./data/images/neural-style/test3.jpg")
# 断言判断图片尺寸是否相等
assert style_img.size() == content_img.size(), \
	"we need to import style and content images of the same size"

# # -------使用plt显示图片
# 转换成PIL格式的图片
unloader = transforms.ToPILImage()
def imshow(tensor, title=None):
	# 把gpu上的张量克隆到cpu
	image = tensor.cpu().clone()
	# remove the fake batch dimension
	# 移除假批处理维度
	image = image.squeeze(0)
	# 把张量转换为pil格式
	image = unloader(image)
	plt.imshow(image)
	if title is not None:
		plt.title(title)
	plt.pause(0.001)
# plt.figure()
# imshow(style_img, title='Style Image')
# plt.figure()
# imshow(content_img, title='Content Image')
# plt.show()

# -------定义内容loss
# 它不是真正的PyTorch的loss函数
class ContentLoss(nn.Module):
	def __init__(self, target):
		super(ContentLoss, self).__init__()
		self.target = target.detach()
	def forward(self, input):
		self.loss = F.mse_loss(input, self.target)
		return input

# -------定义Gram矩阵,Gram=FF.t
def gram_matrix(input):
	# a=batch, b=number of feature maps,
	# (c, d)=dimension of a  f. map(N=c*d)
	a, b, c, d = input.size()
	# 把输入的大小进行调整，这里的features即为F
	features = input.view(a*b, c*d)
	G = torch.mm(features, features.t())
	# 对矩阵G进行标准化
	return G.div(a * b * c * d)

# -------定义风格loss
class StyleLoss(nn.Module):
	def __init__(self, target_feature):
		super(StyleLoss, self).__init__()
		# 返回一个从当前图中分离出来的新张量。新张量不需要梯度
		self.target = gram_matrix(target_feature).detach()
	def forward(self, input):
		G = gram_matrix(input)
		# 即G需要计算梯度，self.target不需要
		self.loss = F.mse_loss(G, self.target)
		# 相当于修改input(暂时这么认为)
		return input

# -------取得vgg19预训练模型的features,并设置为评估模式
# 模型储存位置为C:\Users\GXUCG/.torch\models
cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# -------规范化输入图片
class Normalization(nn.Module):
	def __init__(self, mean, std):
		super(Normalization, self).__init__()
		# 对于某批次的张量，他的shape为[b*c*h*w]
		self.mean = torch.tensor(mean).view(-1, 1, 1)
		self.std = torch.tensor(std).view(-1, 1, 1)
	def forward(self, img):
		return (img - self.mean)/self.std

# -------创建新的顺序模型，包好内容loss和样式loss
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_loss(cnn, normalization_mean, normalization_std,
							style_img, content_img, 
							content_layers = content_layers_default,
							style_layers = style_layers_default):
	cnn = copy.deepcopy(cnn)
	normalization = Normalization(normalization_mean, normalization_std).to(device)
	content_losses = []
	style_losses = []
	# model第一层为标准化输入
	model = nn.Sequential(normalization)
	i = 0
	# 遍历vgg19模型
	for layer in cnn.children():
		# 判断该层是否是Conv2d
		if isinstance(layer, nn.Conv2d):
			i += 1
			name = 'conv_{}'.format(i)
		elif isinstance(layer, nn.ReLU):
			name = 'relu_{}'.format(i)
			layer = nn.ReLU(inplace = False)
		elif isinstance(layer, nn.MaxPool2d):
			name = 'pool_{}'.format(i)
		elif isinstance(layer, nn.BatchNorm2d):
			name = 'bn_{}'.format(i)
		else:
			raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
		# 把该层添加到model中
		model.add_module(name, layer)
		# 判断是否添加内容loss和样式loss
		if name in content_layers:
			# 得到内容图片的特征并设置为不需要求导
			target = model(content_img).detach()
			content_loss = ContentLoss(target)
			model.add_module("content_loss_{}".format(i), content_loss)
			content_losses.append(content_loss)
		if name in style_layers:
			# 得到风格图片的特征并设置为不需要求导
			target_feature = model(style_img).detach()
			style_loss = StyleLoss(target_feature)
			model.add_module("style_loss_{}".format(i), style_loss)
			style_losses.append(style_loss)
	# 修剪图层直到最后一层为内容loss或样式loss
	# 从图的后往前遍历
	for i in range(len(model) - 1, -1, -1):
		if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
			break
	model = model[:(i + 1)]
	return model, style_losses, content_losses

# -------选择输入的图片，可以是内容图片，也可以是白噪声
# 这句是使用内容图片
input_img = content_img.clone()
# # 这句是使用白噪声
# input_img = torch.randn(content_img.data.size(), device=device)
# plt.figure()
# imshow(input_img, title='Input Image')
# plt.show()

# -------设置梯度下降算法
def get_input_optimizer(input_img):
	optimizer = optim.LBFGS([input_img.requires_grad_()])
	return optimizer

# -------运行风格转换函数
def run_style_transfer(cnn, normalization_mean, normalization_std,
					content_img, style_img, input_img, num_steps=300,
					style_weight=1000000, content_weight=1):
	print('Building the style transfer model..')
	# 取得模型和内容，风格loss
	model, style_losses, content_losses = get_style_model_and_loss(cnn,
		normalization_mean, normalization_std, style_img, content_img)
	# 输出模型
	print(model)
	model.to(device)
	# 取得优化器
	optimizer = get_input_optimizer(input_img)
	print('Optimizing...')
	run = [0]
	while run[0] <= num_steps:
		def closure():
			input_img.data.clamp_(0, 1)
			optimizer.zero_grad()
			# 把图像输入模型
			# 此时，在model中的所有模型的forward函数会被依次调用
			model(input_img)

			style_score = 0
			content_score = 0
			for sl in style_losses:
				style_score += sl.loss
			for cl in content_losses:
				content_score += cl.loss
			style_score *= style_weight
			content_score *= content_weight

			loss = style_score + content_score
			loss.backward()

			run[0] += 1
			if run[0] % 50 == 0:
				print("run {}:".format(run))
				print('Style Loss : {:4f} Content Loss: {:4f}'.format(
					style_score, content_score))
				print()
			return style_score + content_score
		# LBFGS优化器的step必须要有个返回loss的闭包作为参数
		optimizer.step(closure)
	input_img.data.clamp_(0, 1)
	return input_img

# -------运行
output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
	content_img, style_img, input_img)
plt.figure()
imshow(output, title='Output Image')

# sphinx_gallery_thumbnail_number = 4
# plt.ioff()
plt.show()
