#!/usr/bin/env bash
python create_coco_tf_record.py --logtostderr --train_image_dir="images/train" --val_image_dir="images/val" --test_image_dir="images/val" --train_annotations_file="images/train/thermal_annotations.json" --val_annotations_file="images/val/thermal_annotations.json" --testdev_annotations_file="images/val/thermal_annotations.json" --output_dir="annotations"
