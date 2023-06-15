from reference.detection.torchutils import draw_bounding_boxes,save_image
import onnx
import torch
import onnxruntime as ort
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

onnx_model = onnx.load("exported_models/ssdlite_mobilenet_v3_100epochs.onnx")
# Check that the IR is well formed
onnx.checker.check_model(onnx_model)
# Print a human readable representation of the graph
onnx.helper.printable_graph(onnx_model.graph)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


img = Image.open("resources/sample_2.jpeg")
# (640, 512)
resize = transforms.Resize([320, 320])
img = resize(img)
# (320,320)
to_tensor = transforms.ToTensor()
img_tensor = to_tensor(img)
# torch.Size([1, 320, 320])
img_tensor.unsqueeze_(0)
# torch.Size([1, 1, 320, 320])
img_tensor = torch.tile(img_tensor, (3, 1, 1))
# torch.Size([1, 3, 320, 320])


ort_session = ort.InferenceSession(
    'exported_models/ssdlite_mobilenet_v3_100epochs.onnx')

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_tensor)}
ort_outs = ort_session.run(None, ort_inputs)

print(ort_outs[0])

# TODO disegnare le bbox sull'immagine
bbox_idx = 0
conf_idx = 1
cat_idx = 2
threshold = 0.5
image_path = "resources/inference_result.jpeg"

# Filter output
conf_gr_than_th = ort_outs[conf_idx] > threshold
bboxes = ort_outs[bbox_idx][np.where(conf_gr_than_th)]
#categories = ort_outs[cat_idx][np.where(conf_gr_than_th)]

# Convert arrays as drawing function wants
bboxes = torch.tensor(bboxes)
# Draw
img_out = draw_bounding_boxes((to_tensor(img)*255).to(torch.uint8),bboxes)
# Save
converter = transforms.ToPILImage(mode=None)
img_test = converter(img_out)
img_test.save(image_path)
print("Image saved as {}".format(image_path))

print("Exported model has been tested with ONNXRuntime, and the result looks good!")
