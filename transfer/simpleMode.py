
import torch
import torch.nn as nn
from torch.autograd import Variable

class simpleMode(nn.Module):
	def __init__(self):
		super().__init__()
		self.layer = self.ensemble()

	def forward(self, x):
		x = self.layer(x)
		return x

	def ensemble(self):
		return nn.Sequential(
			nn.Conv2d(3, 10, 5, padding=2),
			nn.Conv2d(10, 3, 5, padding=2)
			)



inputs = Variable(torch.randn(1,3,100,100))

model = simpleMode()

torch.onnx.export(model, inputs, "simpleMode.proto", verbose=True)

