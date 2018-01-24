import os
import numpy as np
import torch
from torch.autograd import Variable
import utils
from simple_transfer import Net
from utils import StyleLoader, StyleLoaderImage
import skvideo.io
import numpy

def run_demo(mirror=False):
	style_model = Net()
	style_model.load_state_dict(torch.load("./models/Final.model"))
	style_model.eval()
	style_model.cuda()


	style_image = "images/9styles/feathers.jpg"
	style_size = 512

	style = utils.tensor_load_rgbimage(style_image, size=style_size)
	style = style.unsqueeze(0)    
	style = utils.preprocess_batch(style).cuda()


	inputparameters = {}
	outputparameters = {}
	reader = skvideo.io.FFmpegReader("./data/input.mp4",
	                inputdict=inputparameters,
	                outputdict=outputparameters)

	shape = reader.getShape()
	print(shape)
	size = 100
	video = numpy.empty(shape=(size,shape[1], shape[2], shape[3]))

	i = 0
	for img in reader.nextFrame():
		img = img.transpose(2, 0, 1)
		style_v = Variable(style, volatile=True)
		style_model.setTarget(style_v)

		img=torch.from_numpy(img).unsqueeze(0).float()
		img=img.cuda()

		img = Variable(img, volatile=True)
		img = style_model(img)

		img = img.cpu().clamp(0, 255).transpose(1,3).data[0].transpose(0,1).numpy()
		print(img.shape)
		video[i,:,:,:] = img

		i = i+1
		if(i==size):
			break

	skvideo.io.vwrite("./data/outputvideo.mp4", video)


from timeit import default_timer
def main():
	# getting things ready
	end = default_timer()

	# run demo
	run_demo()

	print(default_timer()-end)

if __name__ == '__main__':
	main()
