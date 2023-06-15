import os
import numpy as np
import torch
from PIL import Image
import json
from torch._C import dtype

import torchvision
from torchvision import transforms

import reference.detection.transforms as T
import reference.detection.engine as engine
import matplotlib.pyplot as plt

import reference.detection.torchutils as torchutils
from dataset_utils import get_annotated_images_index_map, get_not_annotated_images_idx, get_label_map


class FlirDataset(torch.utils.data.Dataset):

    def __init__(self, root, transforms):

        self.root = root
        self.transforms = transforms

        # Get images list
        self.imgs = list(
            sorted(os.listdir(os.path.join(root, "thermal_8_bit"))))

        # Retrive only annotated images
        path_json = os.path.join(self.root, "thermal_annotations.json")
        not_annotated_imgs_idx = get_not_annotated_images_idx(
            path_json)
        self.annotated_images_idx_map = get_annotated_images_index_map(
            self.imgs, not_annotated_imgs_idx)

        # Label map
        path_json = "resources/label_map.json"
        self.label_map = get_label_map(path_json)

        # Load annotation file
        path_json = os.path.join(self.root, "thermal_annotations.json")
        f = open(path_json,)
        self.annotations_file = json.load(f)

    def __getitem__(self, index):

        index = self.annotated_images_idx_map[index]

        boxes = []
        labels = []
        area = []
        isCrowd = []
        target = {}
        target["image_id"] = torch.tensor([index])

        img_path = os.path.join(self.root, 'thermal_8_bit', self.imgs[index])
        img = Image.open(img_path).convert('RGB')

        for info in self.annotations_file['annotations']:

            if info['image_id'] == index:
                x_min = info['bbox'][0]
                y_min = info['bbox'][1]
                x_max = x_min + info['bbox'][2]
                y_max = y_min + info['bbox'][3]
                boxes.append(list((x_min, y_min, x_max, y_max)))
                labels.append(self.label_map[info['category_id']])
                area.append(info['area'])
                isCrowd.append(info['iscrowd'])

            # convert everything into a torch.Tensor
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)

            target["labels"] = torch.as_tensor(labels, dtype=torch.int64)

            target["iscrowd"] = torch.zeros(
                (len(labels),), dtype=torch.int64)

            target["area"] = torch.as_tensor(area, dtype=torch.float32)

            if info['image_id'] == index+1:
                break

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.annotated_images_idx_map)


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomIoUCrop())
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


if __name__ == "__main__":

    # Test of flipped image
    print("\nFlipped image test")
    flip = get_transform(train=True)
    no_flip = get_transform(train=False)
    converter = transforms.ToPILImage(mode=None)

    flipped_dataset = FlirDataset("images/train", flip)
    dataset = FlirDataset("images/train", no_flip)
    img_index = 0
    image_path = "test/flipped.jpeg"
    img, target = flipped_dataset.__getitem__(img_index)

    if target is not None:
        img_wbb = torchutils.draw_bounding_boxes(
            (img*255).to(torch.uint8), target["boxes"])
        img_test = converter(img_wbb)
        img_test.save(image_path)
        print("image saved as {}".format(image_path))

        print("Target:")
        for k, v in target.items():
            print("{0}: {1}".format(k, v))

    # Image not flipped
    image_path = "test/not_flipped.jpeg"
    img, target = dataset.__getitem__(img_index)

    if target is not None:
        img_wbb = torchutils.draw_bounding_boxes(
            (img*255).to(torch.uint8), target["boxes"])
        img_test = converter(img_wbb)
        # img_test.show()
        img_test.save(image_path)
        print("image saved as {}".format(image_path))

        print("Target:")
        for k, v in target.items():
            print("{0}: {1}".format(k, v))

    # Image without annotations are special
    print("\nImage without annotation test")
    img_index = 22
    image_path = "test/not_annotated.jpeg"
    img, target = dataset.__getitem__(img_index)

    if target is not None:
        img_wbb = torchutils.draw_bounding_boxes(
            (img*255).to(torch.uint8), target["boxes"])
        img_test = converter(img_wbb)
        # img_test.show()
        img_test.save(image_path)
        print("image saved as {}".format(image_path))

        print("Target:")
        for k, v in target.items():
            print("{0}: {1}".format(k, v))

    # Last image test
    print("\nLast image test")
    img_index = dataset.__len__()-1
    print("Dataset lenght: {0}".format(dataset.__len__()))
    image_path = "test/last.jpeg"
    img, target = dataset.__getitem__(img_index)

    if target is not None:
        img_wbb = torchutils.draw_bounding_boxes(
            (img*255).to(torch.uint8), target["boxes"])
        img_test = converter(img_wbb)
        # img_test.show()
        img_test.save(image_path)
        print("image saved as {}".format(image_path))

        print("Target:")
        for k, v in target.items():
            print("{0}: {1}".format(k, v))
