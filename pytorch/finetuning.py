from torch import optim
import torchvision
import torch
from flir_dataset import FlirDataset, get_transform
from reference.detection.engine import train_one_epoch, evaluate
import reference.detection.utils as utils
import torch

import torch.onnx


def train():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    # use our dataset and defined transformations
    dataset = FlirDataset('images/train',
                             get_transform(train=True))

    dataset_test = FlirDataset(
        'images/val', get_transform(train=False))

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
        num_classes=5, pretrained_backbone=True, trainable_backbone_layers=5)

    model.to(device)

    num_epochs = 100

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.15,
                                momentum=0.9, weight_decay=0.00004)                  
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                    T_max=num_epochs)


    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader,
                                       device, epoch, print_freq=10)

        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)


    # Exporting

    # The mode of the model need to change befor export
    model.eval()
    # Input to the model
    batch_size = 1
    # ONNX exporter need an input to feed the model
    x = torch.randn(batch_size, 3, 320, 320)
    # Input must be sent to GPU as the model
    x = x.to(device)

    # Export the model
    torch.onnx.export(model,               # model being run
                    # model input (or a tuple for multiple inputs)
                    x,
                    # where to save the model (can be a file or file-like object)
                    "exported_models/my_model.onnx",
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=13,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names=['input'],   # the model's input names
                    output_names=['output'],  # the model's output names
                    dynamic_axes={'input': {0: 'batch_size'},    # variable length axes
                                    'output': {0: 'batch_size'}})

    print("That's it!")


if __name__ == "__main__":
    train()
