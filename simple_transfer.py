import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable

class Inspiration(nn.Module):
	""" Inspiration Layer (from MSG-Net paper)
	tuning the featuremap with target Gram Matrix
	ref https://arxiv.org/abs/1703.06953
	"""
	def __init__(self, C, B=1):
		super().__init__()
		# B is equal to 1 or input mini_batch
		self.weight = nn.Parameter(torch.Tensor(1,C,C), requires_grad=True)
		# non-parameter buffer
		self.G = Variable(torch.Tensor(B,C,C), requires_grad=True)
		self.C = C
		self.reset_parameters()

	def reset_parameters(self):
		self.weight.data.uniform_(0.0, 0.02)

	def setTarget(self, target):
		self.G = target

	def forward(self, X):
		# input X is a 3D feature map
		self.P = torch.bmm(self.weight.expand_as(self.G),self.G)
		x = self.P.transpose(1,2)
		stack = [x for y in range(X.size(0))]
		x = torch.cat(stack, 0)
		x = torch.bmm(x, X.view(X.size(0),X.size(1),-1)).view_as(X)
		return x

	def __repr__(self):
		return self.__class__.__name__ + '(' \
			+ 'N x ' + str(self.C) + ')'




class GramMatrix(nn.Module):
	def forward(self, y):
		(b, ch, h, w) = y.size()
		features = y.view(b, ch, w * h)
		features_t = features.transpose(1, 2)
		gram = features.bmm(features_t) / (ch * h * w)
		return gram


class Net(nn.Module):
	def __init__(self, input_nc=3, output_nc=3, ngf=128, norm_layer=nn.BatchNorm2d, n_blocks=6):
		super(Net, self).__init__()
		self.gram = GramMatrix()

		block = Bottleneck
		upblock = UpBottleneck
		expansion = 4

		model1 = []
		model1 += [ConvLayer(input_nc, 64, kernel_size=7, stride=1),
							norm_layer(64),
							nn.ReLU(inplace=True),
							block(64, 32, 2, 1, norm_layer),
							block(32*expansion, ngf, 2, 1, norm_layer)]
		self.model1 = nn.Sequential(*model1)

		model = []
		self.ins = Inspiration(ngf*expansion)
		model += [self.model1]
		model += [self.ins]    

		for i in range(n_blocks):
			model += [block(ngf*expansion, ngf, 1, None, norm_layer)]
		
		model += [upblock(ngf*expansion, 32, 2, norm_layer),
							upblock(32*expansion, 16, 2, norm_layer),
							norm_layer(16*expansion),
							nn.ReLU(inplace=True),
							ConvLayer(16*expansion, output_nc, kernel_size=7, stride=1)]

		self.model = nn.Sequential(*model)

	def setTarget(self, Xs):
		F = self.model1(Xs)
		G = self.gram(F)
		self.ins.setTarget(G)

	def forward(self, input):
		return self.model(input)

import numpy as np
class ConvLayer(torch.nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride):
		super(ConvLayer, self).__init__()
		reflection_padding = int(np.floor(kernel_size / 2))
		self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
		self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

	def forward(self, x):
		out = self.reflection_pad(x)
		out = self.conv2d(out)
		return out


class Bottleneck(nn.Module):
	""" Pre-activation residual block
	Identity Mapping in Deep Residual Networks
	ref https://arxiv.org/abs/1603.05027
	"""
	def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):
		super(Bottleneck, self).__init__()
		self.expansion = 4
		self.downsample = downsample
		if self.downsample is not None:
			self.residual_layer = nn.Conv2d(inplanes, planes * self.expansion,
														kernel_size=1, stride=stride)
		conv_block = []
		conv_block += [norm_layer(inplanes),
									nn.ReLU(inplace=True),
									nn.Conv2d(inplanes, planes, kernel_size=1, stride=1)]
		conv_block += [norm_layer(planes),
									nn.ReLU(inplace=True),
									ConvLayer(planes, planes, kernel_size=3, stride=stride)]
		conv_block += [norm_layer(planes),
									nn.ReLU(inplace=True),
									nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1)]
		self.conv_block = nn.Sequential(*conv_block)
		
	def forward(self, x):
		if self.downsample is not None:
			residual = self.residual_layer(x)
		else:
			residual = x
		return residual + self.conv_block(x)


class UpBottleneck(nn.Module):
	""" Up-sample residual block (from MSG-Net paper)
	Enables passing identity all the way through the generator
	ref https://arxiv.org/abs/1703.06953
	"""
	def __init__(self, inplanes, planes, stride=2, norm_layer=nn.BatchNorm2d):
		super(UpBottleneck, self).__init__()
		self.expansion = 4
		self.residual_layer = UpsampleConvLayer(inplanes, planes * self.expansion,
													  kernel_size=1, stride=1, upsample=stride)
		conv_block = []
		conv_block += [norm_layer(inplanes),
									nn.ReLU(inplace=True),
									nn.Conv2d(inplanes, planes, kernel_size=1, stride=1)]
		conv_block += [norm_layer(planes),
									nn.ReLU(inplace=True),
									UpsampleConvLayer(planes, planes, kernel_size=3, stride=1, upsample=stride)]
		conv_block += [norm_layer(planes),
									nn.ReLU(inplace=True),
									nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1)]
		self.conv_block = nn.Sequential(*conv_block)

	def forward(self, x):
		return  self.residual_layer(x) + self.conv_block(x)



class UpsampleConvLayer(torch.nn.Module):
	"""UpsampleConvLayer
	Upsamples the input and then does a convolution. This method gives better results
	compared to ConvTranspose2d.
	ref: http://distill.pub/2016/deconv-checkerboard/
	"""

	def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
		super(UpsampleConvLayer, self).__init__()
		self.upsample = upsample
		if upsample:
			self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
		self.reflection_padding = int(np.floor(kernel_size / 2))
		if self.reflection_padding != 0:
			self.reflection_pad = nn.ReflectionPad2d(self.reflection_padding)
		self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

	def forward(self, x):
		if self.upsample:
			x = self.upsample_layer(x)
		if self.reflection_padding != 0:
			x = self.reflection_pad(x)
		out = self.conv2d(x)
		return out

############################################################################
# vgg is not directly related with Net module
############################################################################

class Vgg16(torch.nn.Module):
	def __init__(self):
		super(Vgg16, self).__init__()
		self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
		self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

		self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
		self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

		self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
		self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
		self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

		self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
		self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
		self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

		self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
		self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
		self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

	def forward(self, X):
		h = F.relu(self.conv1_1(X))
		h = F.relu(self.conv1_2(h))
		relu1_2 = h
		h = F.max_pool2d(h, kernel_size=2, stride=2)

		h = F.relu(self.conv2_1(h))
		h = F.relu(self.conv2_2(h))
		relu2_2 = h
		h = F.max_pool2d(h, kernel_size=2, stride=2)

		h = F.relu(self.conv3_1(h))
		h = F.relu(self.conv3_2(h))
		h = F.relu(self.conv3_3(h))
		relu3_3 = h
		h = F.max_pool2d(h, kernel_size=2, stride=2)

		h = F.relu(self.conv4_1(h))
		h = F.relu(self.conv4_2(h))
		h = F.relu(self.conv4_3(h))
		relu4_3 = h

		return [relu1_2, relu2_2, relu3_3, relu4_3]



if __name__ == '__main__':
	from torch.autograd import Variable
	import utils
	import os, re

	files = os.listdir("./data")
	files = list(filter(lambda x:re.search("jpg", x), files))
	for file in files:

		content_image = "./data/"+file
		content_size = 512
		style_image = "images/9styles/udnie.jpg"
		style_size = 512

		content_image = utils.tensor_load_rgbimage(content_image, size=content_size, keep_asp=True)
		content_image = content_image.unsqueeze(0)
		style = utils.tensor_load_rgbimage(style_image, size=style_size)
		style = style.unsqueeze(0)    
		style = utils.preprocess_batch(style)
		model = Net(ngf=128)
		model.load_state_dict(torch.load("./models/Final.model"))


		style_v = Variable(style, volatile=True)

		content_image = Variable(utils.preprocess_batch(content_image), volatile=True)
		model.setTarget(style_v)

		output = model(content_image)

		print(output.shape)
		utils.tensor_save_bgrimage(output.data[0], "./data/result/"+file.replace(".jpg", "_result.jpg"), 0)
	'''
	with torch.onnx, people could transfer dynamic model into static, like: dynamic getting image size into just loading static images with fixed size
	'''
	# torch.onnx.export(model, inputs, "Net.proto", verbose=True)