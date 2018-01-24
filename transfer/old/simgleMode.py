'''
JUST A TEST MODULE FOR TORCH2COREML MODEL BUILDING UP
'''
import torch
from torch.autograd import Variable


class MODEL(torch.nn.Module):
	def __init__(self):
		super(MODEL, self).__init__() #this is for compatiblity with python 2.7, otherwise, just use super().__init__()
		self.forward_layer = self.layer()
		self.sigmoid = torch.nn.Sigmoid()

	def forward(self, x):
		x = self.forward_layer(x)
		x = self.sigmoid(x)
		return x

	def layer(self):
		return torch.nn.Sequential(
			torch.nn.Linear(10, 1000),
			torch.nn.ReLU(),
			torch.nn.Linear(1000, 10),
			torch.nn.Sigmoid()
		)

if __name__ == '__main__':
	import numpy
	inputs = Variable(torch.from_numpy(numpy.random.randn(10)).float())
	# print(inputs)
	model = MODEL()
	outputs = model(inputs)
	print(outputs)