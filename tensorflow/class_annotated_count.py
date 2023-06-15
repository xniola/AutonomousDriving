"""Verifica le classi del dataset che sono state annotate"""

import json


f = open('images/train/thermal_annotations.json')
data = json.load(f)
category = {}

for i in data['annotations']:
    cat = i['category_id']
    if cat in category.keys():
        category[cat] += 1
    else:
        category[cat] = 1

f.close()

print("classi delle annotazioni in images/train/thermal_annotations.json: ", category)


f = open('images/val/thermal_annotations.json')
data = json.load(f)
category = {}

for i in data['annotations']:
    cat = i['category_id']
    if cat in category.keys():
        category[cat] += 1
    else:
        category[cat] = 1

f.close()

print("classi delle annotazioni in images/val/thermal_annotations.json: ", category)
