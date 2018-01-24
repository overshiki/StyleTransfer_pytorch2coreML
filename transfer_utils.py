from simple_transfer import Net
from torch.autograd import Variable
import utils
import torch


content_image = "./data/face.jpg"
content_size = 512
content_image = utils.tensor_load_rgbimage(content_image, size=content_size, keep_asp=True)
content_image = content_image.unsqueeze(0)
model = Net()
model.load_state_dict(torch.load("./models/Final.model"))


# style_image = "images/9styles/udnie.jpg"
# style_size = 512
# style = utils.tensor_load_rgbimage(style_image, size=style_size)
# style = style.unsqueeze(0)    
# style = utils.preprocess_batch(style)
# style_v = Variable(style, volatile=True)
# model.setTarget(style_v)


content_image = Variable(utils.preprocess_batch(content_image), volatile=True)

# output = model(content_image)

# print(output.shape)
# utils.tensor_save_bgrimage(output.data[0], "./data/result/"+file.replace(".jpg", "_result.jpg"), 0)

torch.onnx.export(model, content_image, "wrapper.proto", verbose=True)