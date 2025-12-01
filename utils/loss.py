import numpy as np
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import *


class yoloLoss(Module):
    def __init__(self, num_class=20):
        super(yoloLoss, self).__init__()
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        self.S = 14
        self.B = 2
        self.C = num_class
        self.step = 1.0 / 14

    def compute_iou(self, box1, box2, index):
        box1 = torch.clone(box1)
        box2 = torch.clone(box2)
        box1 = self.conver_box(box1, index)
        box2 = self.conver_box(box2, index)
        x1, y1, w1, h1 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        x2, y2, w2, h2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
        # 
        inter_w = (w1 + w2) - (torch.max(x1 + w1, x2 + w2) - torch.min(x1, x2))
        inter_h = (h1 + h2) - (torch.max(y1 + h1, y2 + h2) - torch.min(y1, y2))
        inter_h = torch.clamp(inter_h, 0)
        inter_w = torch.clamp(inter_w, 0)
        # 
        inter = inter_w * inter_h
        union = w1 * h1 + w2 * h2 - inter
        return inter / union

    def conver_box(self, box, index):
        i, j = index
        box[:, 0], box[:, 1] = [(box[:, 0] + i) * self.step - box[:, 2] / 2,
                                (box[:, 1] + j) * self.step - box[:, 3] / 2]
        box = torch.clamp(box, 0)
        return box

    def forward(self, pred, target):
        batch_size = pred.size(0)
        target_boxes = target[:, :, :, :10].contiguous().reshape(
            (-1, self.S, self.S, 2, 5))
        pred_boxes = pred[:, :, :, :10].contiguous().reshape(
            (-1, self.S, self.S, 2, 5))
        
        target_cls = target[:, :, :, 10:]
        pred_cls = pred[:, :, :, 10:]
        
        # [수정 1] UserWarning 해결: .byte() -> .bool()로 변경
        obj_mask = (target_boxes[..., 4] > 0).bool()  
        sig_mask = obj_mask[..., 1].bool() 
        index = torch.where(sig_mask == True) 

        for img_i, y, x in zip(*index):
            img_i, y, x = img_i.item(), y.item(), x.item()
            pbox = pred_boxes[img_i, y, x]
            target_box = target_boxes[img_i, y, x]
            ious = self.compute_iou(pbox[:, :4], target_box[:, :4], [x, y])
            iou, max_i = ious.max(0)
            obj_mask[img_i, y, x, 1 - max_i] = 0
        
        noobj_mask = ~obj_mask
         
        noobj_loss = F.mse_loss(pred_boxes[noobj_mask][:, 4],
                                target_boxes[noobj_mask][:, 4],
                                reduction="sum")
        obj_loss = F.mse_loss(pred_boxes[obj_mask][:, 4],
                              target_boxes[obj_mask][:, 4],
                              reduction="sum")
        xy_loss = F.mse_loss(pred_boxes[obj_mask][:, :2],
                             target_boxes[obj_mask][:, :2],
                             reduction="sum")
        
        wh_loss = F.mse_loss(torch.sqrt(target_boxes[obj_mask][:, 2:4]),
                             torch.sqrt(pred_boxes[obj_mask][:, 2:4]),
                             reduction="sum")
        
        class_loss = F.mse_loss(pred_cls[sig_mask],
                                target_cls[sig_mask],
                                reduction="sum")

        loss = obj_loss + self.lambda_noobj * noobj_loss \
                    + self.lambda_coord * xy_loss + self.lambda_coord * wh_loss \
                    + class_loss
        
        # [수정 2] main.py의 로깅을 위해 상세 loss 값들도 함께 리턴 (Batch Size로 나누어 정규화)
        return loss/batch_size, xy_loss/batch_size, wh_loss/batch_size, obj_loss/batch_size, noobj_loss/batch_size, class_loss/batch_size