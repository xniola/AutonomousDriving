import json
import pprint
from typing import List

PATH_OF_IMAGES = "images/train/thermal_8_bit"
PATH_OF_ANNOTATIONS = "images/train/thermal_annotations.json"


def count_categories(path_to_json):

    f = open(path_to_json)
    j = json.load(f)

    categories = {}

    for ann in j["annotations"]:
        cat = str(ann["category_id"])

        if cat in categories.keys():
            categories[cat] += 1
        else:
            categories[cat] = 1

    return categories


def get_not_annotated_images(path_to_json):

    f = open(path_to_json)
    j = json.load(f)

    img_ann_ids = []
    img_ids = []
    imgs = []
    not_annotated = []

    for ann in j["annotations"]:
        img_ann_ids.append(ann["image_id"])

    for img_info in j["images"]:
        img_ids.append(img_info["id"])
        imgs.append((img_info["id"], img_info["file_name"]))

    for img_id, img_name in imgs:
        if img_id not in img_ann_ids:
            not_annotated.append(img_name)

    return not_annotated


def get_not_annotated_images_idx(path_to_json):

    f = open(path_to_json)
    j = json.load(f)

    img_ann_ids = []
    img_ids = []
    imgs = []
    not_annotated = []

    for ann in j["annotations"]:
        img_ann_ids.append(ann["image_id"])

    for img_info in j["images"]:
        img_ids.append(img_info["id"])
        imgs.append((img_info["id"], img_info["file_name"]))

    for img_id, img_name in imgs:
        if img_id not in img_ann_ids:
            not_annotated.append(img_id)

    return not_annotated


def get_label_map(path_to_json):
    f = open(path_to_json)
    j = json.load(f)

    label_map = {}
    for i, item in enumerate(j):
        label_map[item["id"]] = i

    return label_map


def get_annotated_images_index_map(images: List, indices_to_remove: List):
    idx_map = {}
    dataset_idx = 0
    for img_idx, _ in enumerate(images):
        if img_idx not in indices_to_remove:   
            idx_map[dataset_idx] = img_idx
            dataset_idx += 1

    return idx_map


if __name__ == "__main__":

    categories = count_categories(PATH_OF_ANNOTATIONS)

    print("\nCategories:\n")
    pprint.pprint(categories)
    print()

    not_annotated_images = get_not_annotated_images(PATH_OF_ANNOTATIONS)

    print("Not annotated images ({0}):\n".format(len(not_annotated_images)))
    pprint.pprint(not_annotated_images)
    print()
