import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from albumentations.augmentations.crops.functional import center_crop


__all__ = ['Evaluator', 'train', 'validation']


class Evaluator():
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def reset(self):
        self.count = np.zeros(self.num_classes, dtype=np.int32)
        self.iou = np.zeros(self.num_classes, dtype=np.float32)
        self.acc = np.zeros(self.num_classes, dtype=np.float32)

    def predict_masks(self, outputs):
        pred_masks = torch.argmax(F.softmax(outputs, dim=1), dim=1)
        return pred_masks

    def evaluate(self, data_loader, model, device):
        running_miou = []
        running_macc = []
        tbar = tqdm(data_loader, desc="evaluation steps", total=len(data_loader))

        model.eval()
        with torch.no_grad():
            for images, gt_masks, (origin_heights, origin_widths) in tbar:
                images, gt_masks = images.to(device), gt_masks.to(device).long()

                score_masks = model(images)
                pred_masks = self.predict_masks(score_masks)

                for pd_msk, gt_msk, h, w in zip(pred_masks.cpu().numpy().astype(np.uint8),
                                                gt_masks.cpu().numpy().astype(np.uint8),
                                                origin_heights,
                                                origin_widths):
                    max_size = max(h.item(), w.item())

                    if pd_msk.shape[-1] != max_size:
                        pd_msk = cv2.resize(pd_msk, (max_size, max_size), interpolation=cv2.INTER_NEAREST)
                        gt_msk = cv2.resize(gt_msk, (max_size, max_size), interpolation=cv2.INTER_NEAREST)

                    pd_msk = center_crop(pd_msk, h, w)
                    gt_msk = center_crop(gt_msk, h, w)

                    class_list = np.unique(gt_msk)
                    for class_idx in class_list:
                        intersection = ((class_idx == gt_msk) & (class_idx == pd_msk)).astype(np.float32).sum()
                        union = ((class_idx == gt_msk) | (class_idx == pd_msk)).astype(np.float32).sum()
                        gt_pixel = (class_idx == gt_msk).astype(np.float32).sum()

                        self.iou[class_idx] += intersection / union
                        self.acc[class_idx] += intersection / gt_pixel
                        self.count[class_idx] += 1

    def get_metrics(self):
        iou = self.iou / self.count
        pixel_acc = self.acc / self.count

        mean_iou = np.mean(iou)
        mean_acc = np.mean(pixel_acc)

        return iou, pixel_acc, mean_iou, mean_acc


def train(data_loader, model, loss_fn, optimizer, device):
    running_loss = []
    tbar = tqdm(data_loader, desc="train steps", total=len(data_loader))
    
    model.train()
    for images, gt_masks in tbar:
        images, gt_masks = images.to(device), gt_masks.to(device).long()

        optimizer.zero_grad()
        score_masks = model(images)
        loss = loss_fn(score_masks, gt_masks)
        loss.backward()
        optimizer.step()
        
        running_loss.append(loss.item())
            
    return np.mean(running_loss)


def validation(data_loader, model, loss_fn, device):
    running_loss = []
    tbar = tqdm(data_loader, desc="valid steps", total=len(data_loader))
    
    model.eval()
    with torch.no_grad():
        for images, gt_masks in tbar:
            images, gt_masks = images.to(device), gt_masks.to(device).long()

            score_masks = model(images)
            loss = loss_fn(score_masks, gt_masks)

            running_loss.append(loss.item())
            
    return np.mean(running_loss)