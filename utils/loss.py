import torch
import torch.nn as nn
import torch.nn.functional as F

class yoloLoss(nn.Module):
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
        
        inter_w = (w1 + w2) - (torch.max(x1 + w1, x2 + w2) - torch.min(x1, x2))
        inter_h = (h1 + h2) - (torch.max(y1 + h1, y2 + h2) - torch.min(y1, y2))
        inter_h = torch.clamp(inter_h, 0)
        inter_w = torch.clamp(inter_w, 0)
        
        inter = inter_w * inter_h
        union = w1 * h1 + w2 * h2 - inter + 1e-6 # [수정] 0으로 나누기 방지 (epsilon 추가)
        return inter / union

    def conver_box(self, box, index):
        i, j = index
        box[:, 0], box[:, 1] = [(box[:, 0] + i) * self.step - box[:, 2] / 2,
                                (box[:, 1] + j) * self.step - box[:, 3] / 2]
        box = torch.clamp(box, 0)
        return box

    def forward(self, pred, target):
        batch_size = pred.size(0)
        
        target_boxes = target[:, :, :, :10].contiguous().reshape((-1, self.S, self.S, 2, 5))
        pred_boxes = pred[:, :, :, :10].contiguous().reshape((-1, self.S, self.S, 2, 5))
        
        target_cls = target[:, :, :, 10:]
        pred_cls = pred[:, :, :, 10:]
        
        obj_mask = (target_boxes[..., 4] > 0).byte()
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
         
        # --- Loss Components Calculation ---
        noobj_loss = F.mse_loss(pred_boxes[noobj_mask][:, 4],
                                target_boxes[noobj_mask][:, 4],
                                reduction="sum")
        
        obj_loss = F.mse_loss(pred_boxes[obj_mask][:, 4],
                              target_boxes[obj_mask][:, 4],
                              reduction="sum")
        
        xy_loss = F.mse_loss(pred_boxes[obj_mask][:, :2],
                             target_boxes[obj_mask][:, :2],
                             reduction="sum")
        
        # [수정] sqrt 씌우기 전에 음수 방지 (절댓값 + epsilon)
        # 1. 0보다 작은 값이 들어오면 sqrt에서 NaN 발생 -> torch.abs() 또는 clamp 사용
        # 2. 아주 작은 값(1e-6)을 더해서 0이 되는 것도 방지
        wh_loss = F.mse_loss(torch.sqrt(torch.clamp(target_boxes[obj_mask][:, 2:4], min=1e-6)),
                             torch.sqrt(torch.clamp(pred_boxes[obj_mask][:, 2:4], min=1e-6)),
                             reduction="sum")
        
        class_loss = F.mse_loss(pred_cls[sig_mask],
                                target_cls[sig_mask],
                                reduction="sum")

        # Total Loss
        loss = obj_loss + self.lambda_noobj * noobj_loss \
                    + self.lambda_coord * xy_loss + self.lambda_coord * wh_loss \
                    + class_loss
        
        loss_dict = {
            'loss_xy': (self.lambda_coord * xy_loss).item() / batch_size,
            'loss_wh': (self.lambda_coord * wh_loss).item() / batch_size,
            'loss_obj': obj_loss.item() / batch_size,
            'loss_noobj': (self.lambda_noobj * noobj_loss).item() / batch_size,
            'loss_class': class_loss.item() / batch_size,
            'loss_total': loss.item() / batch_size
        }
        
        return loss / batch_size, loss_dict