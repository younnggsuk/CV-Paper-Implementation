import os
import cv2
import numpy as np
from PIL import Image


VOC_COLORMAP = [
    [0, 0, 0],          # 0.  background
    [128, 0, 0],        # 1.  aeroplane
    [0, 128, 0],        # 2.  bicycle
    [128, 128, 0],      # 3.  bird
    [0, 0, 128],        # 4.  boat
    [128, 0, 128],      # 5.  bottle
    [0, 128, 128],      # 6.  bus
    [128, 128, 128],    # 7.  car
    [64, 0, 0],         # 8.  cat
    [192, 0, 0],        # 9.  chair
    [64, 128, 0],       # 10. cow
    [192, 128, 0],      # 11. diningtable
    [64, 0, 128],       # 12. dog
    [192, 0, 128],      # 13. horse
    [64, 128, 128],     # 14. motorbike
    [192, 128, 128],    # 15. person
    [0, 64, 0],         # 16. potted plant
    [128, 64, 0],       # 17. sheep
    [0, 192, 0],        # 18. sofa
    [128, 192, 0],      # 19. train
    [0, 64, 128]        # 20. tv/monitor
]


def make_label_mask(label):
    label_mask = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)
    for cls_index, color_map in enumerate(VOC_COLORMAP):
        label_mask[np.where(np.all(label == color_map, axis=-1))[:]] = cls_index

    return label_mask


if __name__ == "__main__":
    data_dir = "../data/"
    label_dir = os.path.join(data_dir, "VOCdevkit", "VOC2012", "SegmentationClass")
    target_dir = os.path.join(data_dir, "VOCdevkit", "VOC2012", "SegmentationClassAug")

    train_files = []
    with open(os.path.join(data_dir, "VOCdevkit", "VOC2012", "ImageSets", "Segmentation", "train.txt"), "rt") as f:
        for line in f.readlines():
            train_files.append(line.replace('\n', ''))

    val_files = []
    with open(os.path.join(data_dir, "VOCdevkit", "VOC2012", "ImageSets", "Segmentation", "val.txt"), "rt") as f:
        for line in f.readlines():
            val_files.append(line.replace('\n', ''))

    for file_name in train_files:
        src_path = os.path.join(label_dir, file_name + ".png")
        tgt_path = os.path.join(target_dir, file_name + ".png")

        label = cv2.imread(src_path)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        Image.fromarray(make_label_mask(label)).save(tgt_path, 'PNG')
    
    for file_name in val_files:
        src_path = os.path.join(label_dir, file_name + ".png")
        tgt_path = os.path.join(target_dir, file_name + ".png")

        label = cv2.imread(src_path)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        Image.fromarray(make_label_mask(label)).save(tgt_path, 'PNG')