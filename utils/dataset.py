import os
import os.path
import random
import numpy as np
import torch
import torch.utils.data as data
import cv2

class Dataset(data.Dataset):
    image_size = 448

    def __init__(self, root, file_names, train, transform):
        print('DATA INITIALIZATION (Full Option: Mosaic + Mixup + Smoothing)')
        
        self.root_images = os.path.join(root, 'Images')
        self.root_labels = os.path.join(root, 'Labels')
        self.train = train
        self.transform = transform
        self.f_names = []
        self.boxes = []
        self.labels = []
        self.mean = (123, 117, 104)  # RGB
        
        # 제어 스위치
        self.enable_mosaic = True 
        self.enable_mixup = True

        for line in file_names:
            line = line.rstrip()
            txt_path = f"{self.root_labels}/{line}.txt"
            if not os.path.exists(txt_path): continue
            
            with open(txt_path) as f:
                objects = f.readlines()
                self.f_names.append(line + '.jpg')
                box = []
                label = []
                for object in objects:
                    c, x1, y1, x2, y2 = map(float, object.rstrip().split())
                    box.append([x1, y1, x2, y2])
                    label.append(int(c) + 1)
                self.boxes.append(torch.Tensor(box))
                self.labels.append(torch.LongTensor(label))
        self.num_samples = len(self.boxes)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.train:
            # 확률적으로 Mixup, Mosaic, 일반 로드 선택
            prob = random.random()
            
            if self.enable_mixup and prob < 0.15:
                img, boxes, labels = self.mixup(idx)
            elif self.enable_mosaic and prob < (0.15 + 0.5): # 약 42.5% 확률로 Mosaic
                img, boxes, labels = self.mosaic(idx)
            else:
                img, boxes, labels = self.load_image_target(idx)
        else:
            img, boxes, labels = self.load_image_target(idx)

        # 공통 Augmentation
        if self.train:
            img, boxes = self.random_flip(img, boxes)
            img, boxes = self.randomScale(img, boxes)
            img = self.randomBlur(img)
            img = self.RandomBrightness(img)
            img = self.RandomHue(img)
            img = self.RandomSaturation(img)
            img, boxes, labels = self.randomShift(img, boxes, labels)
            img, boxes, labels = self.randomCrop(img, boxes, labels)

        h, w, _ = img.shape
        boxes /= torch.Tensor([w, h, w, h]).expand_as(boxes)
        img = self.subMean(img, self.mean)
        img = cv2.resize(img, (self.image_size, self.image_size))
        
        target = self.encoder(boxes, labels)

        for t in self.transform:
            img = t(img)

        return img, target

    def load_image_target(self, idx):
        img_path = os.path.join(self.root_images, self.f_names[idx])
        img = cv2.imread(img_path)
        if img is None:
            if idx != 0: return self.load_image_target(0)
            else: raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()
        return img, boxes, labels

    # [수정됨] Mixup Augmentation
    def mixup(self, idx):
        idx2 = random.randint(0, self.num_samples - 1)
        
        # 1. 두 이미지 로드
        img1, box1, label1 = self.load_image_target(idx)
        img2, box2, label2 = self.load_image_target(idx2)
        
        # 2. [핵심 수정] 리사이즈 전에 원본 크기 저장 (여기서 에러 났었음)
        h1, w1, _ = img1.shape
        h2, w2, _ = img2.shape
        
        # 3. 448x448로 리사이즈
        img1 = cv2.resize(img1, (448, 448))
        img2 = cv2.resize(img2, (448, 448))
        
        # 4. 이미지 믹스 (Beta 분포)
        lam = np.random.beta(1.5, 1.5)
        img = lam * img1 + (1 - lam) * img2
        img = img.astype(np.uint8) 
        
        # 5. 박스 좌표 스케일링 (원본 크기 -> 448 크기)
        # box1 좌표 변환
        box1[:, [0, 2]] *= (448 / w1)
        box1[:, [1, 3]] *= (448 / h1)

        # box2 좌표 변환
        box2[:, [0, 2]] *= (448 / w2)
        box2[:, [1, 3]] *= (448 / h2)
        
        # 6. 박스와 라벨 합치기
        boxes = torch.cat((box1, box2), 0)
        labels = torch.cat((label1, label2), 0)
        
        return img, boxes, labels

    def mosaic(self, idx):
        min_offset_x = 0.3
        min_offset_y = 0.3
        w, h = 448, 448 
        idxs = [idx] + [random.randint(0, self.num_samples - 1) for _ in range(3)]
        cx = int(w * min_offset_x + random.random() * (w * (1 - 2 * min_offset_x)))
        cy = int(h * min_offset_y + random.random() * (h * (1 - 2 * min_offset_y)))
        bgr = np.zeros((h, w, 3), np.float32)
        bgr[:, :, :] = self.mean 
        boxes_list = []
        labels_list = []
        for i, index in enumerate(idxs):
            img, box, label = self.load_image_target(index)
            h_img, w_img, _ = img.shape
            if i == 0:
                x1a, y1a, x2a, y2a = max(cx - w_img, 0), max(cy - h_img, 0), cx, cy
                x1b, y1b, x2b, y2b = w_img - (x2a - x1a), h_img - (y2a - y1a), w_img, h_img
            elif i == 1:
                x1a, y1a, x2a, y2a = cx, max(cy - h_img, 0), min(cx + w_img, w), cy
                x1b, y1b, x2b, y2b = 0, h_img - (y2a - y1a), min(w_img, x2a - x1a), h_img
            elif i == 2:
                x1a, y1a, x2a, y2a = max(cx - w_img, 0), cy, cx, min(cy + h_img, h)
                x1b, y1b, x2b, y2b = w_img - (x2a - x1a), 0, w_img, min(h_img, y2a - y1a)
            elif i == 3:
                x1a, y1a, x2a, y2a = cx, cy, min(cx + w_img, w), min(cy + h_img, h)
                x1b, y1b, x2b, y2b = 0, 0, min(w_img, x2a - x1a), min(h_img, y2a - y1a)
            bgr[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            pad_w = x1a - x1b
            pad_h = y1a - y1b
            box_new = box.clone()
            box_new[:, 0] += pad_w
            box_new[:, 1] += pad_h
            box_new[:, 2] += pad_w
            box_new[:, 3] += pad_h
            boxes_list.append(box_new)
            labels_list.append(label)
        if len(boxes_list) == 0: return self.load_image_target(idx)
        boxes = torch.cat(boxes_list, 0)
        labels = torch.cat(labels_list, 0)
        boxes[:, 0] = boxes[:, 0].clamp_(min=0, max=w)
        boxes[:, 1] = boxes[:, 1].clamp_(min=0, max=h)
        boxes[:, 2] = boxes[:, 2].clamp_(min=0, max=w)
        boxes[:, 3] = boxes[:, 3].clamp_(min=0, max=h)
        box_w = boxes[:, 2] - boxes[:, 0]
        box_h = boxes[:, 3] - boxes[:, 1]
        valid_mask = (box_w > 10) & (box_h > 10)
        boxes = boxes[valid_mask]
        labels = labels[valid_mask]
        if len(boxes) == 0: return self.load_image_target(idx)
        return bgr, boxes, labels

    def subMean(self, bgr, mean):
        mean = np.array(mean, dtype=np.float32)
        bgr = bgr - mean
        return bgr
    def random_flip(self, im, boxes):
        if random.random() < 0.5:
            im_lr = np.fliplr(im).copy()
            h, w, _ = im.shape
            xmin = w - boxes[:, 2]
            xmax = w - boxes[:, 0]
            boxes[:, 0] = xmin
            boxes[:, 2] = xmax
            return im_lr, boxes
        return im, boxes
    def randomScale(self, bgr, boxes):
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            height, width, c = bgr.shape
            bgr = cv2.resize(bgr, (int(width * scale), height))
            scale_tensor = torch.FloatTensor([[scale, 1, scale, 1]]).expand_as(boxes)
            boxes = boxes * scale_tensor
            return bgr, boxes
        return bgr, boxes
    def randomBlur(self, bgr):
        if random.random() < 0.5:
            bgr = cv2.blur(bgr, (5, 5))
        return bgr
    def RandomBrightness(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            v = v * adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr
    def RandomHue(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            h = h * adjust
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr
    def RandomSaturation(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            s = s * adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr
    def BGR2HSV(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    def HSV2BGR(self, img):
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    def randomShift(self, bgr, boxes, labels):
        center = (boxes[:, 2:] + boxes[:, :2]) / 2
        if random.random() < 0.5:
            height, width, c = bgr.shape
            after_shfit_image = np.zeros((height, width, c), dtype=bgr.dtype)
            after_shfit_image[:, :, :] = (104, 117, 123)  # bgr
            shift_x = random.uniform(-width * 0.2, width * 0.2)
            shift_y = random.uniform(-height * 0.2, height * 0.2)
            if shift_x >= 0 and shift_y >= 0:
                after_shfit_image[int(shift_y):, int(shift_x):, :] = bgr[:height - int(shift_y), :width - int(shift_x),:]
            elif shift_x >= 0 and shift_y < 0:
                after_shfit_image[:height + int(shift_y), int(shift_x):, :] = bgr[-int(shift_y):, :width - int(shift_x),:]
            elif shift_x < 0 and shift_y >= 0:
                after_shfit_image[int(shift_y):, :width + int(shift_x), :] = bgr[:height - int(shift_y), -int(shift_x):,:]
            elif shift_x < 0 and shift_y < 0:
                after_shfit_image[:height + int(shift_y), :width + int(shift_x), :] = bgr[-int(shift_y):, -int(shift_x):,:]

            shift_xy = torch.FloatTensor([[shift_x, shift_y, shift_x, shift_y]]).expand_as(boxes)
            center = center + shift_xy[:, :2]
            mask1 = (center[:, 0] > 0) & (center[:, 0] < width)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < height)
            mask = (mask1 & mask2).view(-1, 1)
            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
            if len(boxes_in) == 0:
                return bgr, boxes, labels
            box_shift = torch.FloatTensor([[shift_x, shift_y, shift_x, shift_y]]).expand_as(boxes_in)
            boxes_in = boxes_in + box_shift
            labels_in = labels[mask.view(-1)]
            return after_shfit_image, boxes_in, labels_in
        return bgr, boxes, labels
    def randomCrop(self, bgr, boxes, labels):
        if random.random() < 0.5:
            center = (boxes[:, 2:] + boxes[:, :2]) / 2
            height, width, c = bgr.shape
            h = random.uniform(0.6 * height, height)
            w = random.uniform(0.6 * width, width)
            x = random.uniform(0, width - w)
            y = random.uniform(0, height - h)
            x, y, h, w = int(x), int(y), int(h), int(w)
            center = center - torch.FloatTensor([[x, y]]).expand_as(center)
            mask1 = (center[:, 0] > 0) & (center[:, 0] < w)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < h)
            mask = (mask1 & mask2).view(-1, 1)
            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
            if len(boxes_in) == 0:
                return bgr, boxes, labels
            box_shift = torch.FloatTensor([[x, y, x, y]]).expand_as(boxes_in)
            boxes_in = boxes_in - box_shift
            boxes_in[:, 0] = boxes_in[:, 0].clamp_(min=0, max=w)
            boxes_in[:, 2] = boxes_in[:, 2].clamp_(min=0, max=w)
            boxes_in[:, 1] = boxes_in[:, 1].clamp_(min=0, max=h)
            boxes_in[:, 3] = boxes_in[:, 3].clamp_(min=0, max=h)
            labels_in = labels[mask.view(-1)]
            img_croped = bgr[y:y + h, x:x + w, :]
            return img_croped, boxes_in, labels_in
        return bgr, boxes, labels

    def encoder(self, boxes, labels):
        grid_num = 14
        num_classes = 20
        target = torch.zeros((grid_num, grid_num, 30))
        cell_size = 1. / grid_num
        wh = boxes[:, 2:] - boxes[:, :2]
        cxcy = (boxes[:, 2:] + boxes[:, :2]) / 2
        
        # Label Smoothing
        smooth_eps = 0.1
        
        for i in range(cxcy.size()[0]):
            cxcy_sample = cxcy[i]
            ij = (cxcy_sample / cell_size).ceil() - 1
            
            target[int(ij[1]), int(ij[0]), 4] = 1
            target[int(ij[1]), int(ij[0]), 9] = 1
            
            target[int(ij[1]), int(ij[0]), 10:] = smooth_eps / num_classes
            class_idx = int(labels[i]) + 9 
            target[int(ij[1]), int(ij[0]), class_idx] += (1.0 - smooth_eps)
            
            xy = ij * cell_size
            delta_xy = (cxcy_sample - xy) / cell_size
            target[int(ij[1]), int(ij[0]), 2:4] = wh[i]
            target[int(ij[1]), int(ij[0]), :2] = delta_xy
            target[int(ij[1]), int(ij[0]), 7:9] = wh[i]
            target[int(ij[1]), int(ij[0]), 5:7] = delta_xy
        return target