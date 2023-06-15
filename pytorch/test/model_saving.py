import torchvision
import torch

import torch.onnx

import pprint

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
    pretrained=True)
model.to(device)

# The mode of the model need to change befor export
model.eval()

# Input to the model
batch_size = 1
# ONNX exporter need an input to feed the model
x = torch.randn(batch_size, 3, 320, 320)
# Input must be sent to GPU as the model
x = x.to(device)

torch_out = model(x)
pprint.pprint(torch_out)

# Export the model
torch.onnx.export(model,               # model being run
                  # model input (or a tuple for multiple inputs)
                  x,
                  # where to save the model (can be a file or file-like object)
                  "test/model.onnx",
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=13,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],   # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},    # variable length axes
                                'output': {0: 'batch_size'}})
