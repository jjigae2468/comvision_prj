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
        print('DATA INITIALIZATION')
        
        self.root_images = os.path.join(root, 'Images')
        self.root_labels = os.path.join(root, 'Labels')
        self.train = train
        self.transform = transform
        self.f_names = []
        self.boxes = []
        self.labels = []
        self.mean = (123, 117, 104)  # RGB

        for line in file_names:
            line = line.rstrip()
            with open(f"{self.root_labels}/{line}.txt") as f:
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

    def __getitem__(self, idx):
        # [수정] 50% 확률로 Mosaic Augmentation 적용
        if self.train and random.random() < 0.5:
            img, boxes, labels = self.load_mosaic(idx)
            # Mosaic 적용 시에는 기하학적 변형(Shift/Scale 등)은 생략하고 색상 노이즈만 추가
            img = self.randomBlur(img)
            img = self.RandomBrightness(img)
            img = self.RandomHue(img)
            img = self.RandomSaturation(img)
        else:
            # 기존 로직 (Mosaic 안 걸렸을 때)
            f_name = self.f_names[idx]
            img = cv2.imread(os.path.join(self.root_images, f_name))
            boxes = self.boxes[idx].clone()
            labels = self.labels[idx].clone()

            if self.train:
                # img = self.random_bright(img)
                img, boxes = self.random_flip(img, boxes)
                img, boxes = self.randomScale(img, boxes)
                img = self.randomBlur(img)
                img = self.RandomBrightness(img)
                img = self.RandomHue(img)
                img = self.RandomSaturation(img)
                img, boxes, labels = self.randomShift(img, boxes, labels)
                img, boxes, labels = self.randomCrop(img, boxes, labels)
        
        h, w, _ = img.shape
        # 박스 정규화 (0~1)
        boxes /= torch.Tensor([w, h, w, h]).expand_as(boxes)
        
        img = self.BGR2RGB(img)
        img = self.subMean(img, self.mean)
        img = cv2.resize(img, (self.image_size, self.image_size))
        
        target = self.encoder(boxes, labels)  # 14x14x30
        for t in self.transform:
            img = t(img)

        return img, target

    def __len__(self):
        return self.num_samples

    # [수정됨] Mosaic Augmentation 구현 (에러 수정 완료)
    def load_mosaic(self, index):
        labels4 = []
        s = self.image_size
        
        # [수정] 중심점(cut) 랜덤 설정 (이전 코드 에러 원인 수정)
        yc = int(random.uniform(s * 0.5, s * 1.5))
        xc = int(random.uniform(s * 0.5, s * 1.5))
        
        # 4장의 이미지 인덱스 선택
        indices = [index] + [random.randint(0, self.num_samples - 1) for _ in range(3)]

        # 빈 캔버스 (회색)
        result_img = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8) 

        for i, idx in enumerate(indices):
            # 이미지 로드
            path = os.path.join(self.root_images, self.f_names[idx])
            img = cv2.imread(path)
            h, w, _ = img.shape
            
            # 박스 로드
            boxes = self.boxes[idx].clone()
            labels = self.labels[idx].clone()

            # 배치 위치 계산
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(yc + h, s * 2)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(h, y2a - y1a)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(yc + h, s * 2)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(h, y2a - y1a)

            # 이미지 붙이기
            result_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            # 박스 좌표 이동
            if len(boxes) > 0:
                boxes[:, 0] += padw
                boxes[:, 2] += padw
                boxes[:, 1] += padh
                boxes[:, 3] += padh
                
                # 라벨 정보 유지 (Label + Box)
                for box, label in zip(boxes, labels):
                    labels4.append(torch.cat([torch.tensor([label]), box]))

        # 결과 이미지 리사이즈 (448x448)
        final_img = cv2.resize(result_img, (s, s))
        
        # 박스 좌표 조정 (Resize 비율에 맞춰서)
        scale = s / (s * 2)  # 0.5
        
        if len(labels4) > 0:
            labels4 = torch.stack(labels4)
            # 좌표 스케일링
            labels4[:, 1:] *= scale
            
            # 캔버스 밖으로 나간 박스 클리핑
            labels4[:, 1:] = labels4[:, 1:].clamp(min=0, max=s-1)
            
            out_boxes = labels4[:, 1:]
            out_labels = labels4[:, 0]
        else:
            out_boxes = torch.zeros((0, 4))
            out_labels = torch.zeros((0,))

        return final_img, out_boxes, out_labels

    def encoder(self, boxes, labels):
        grid_num = 14
        target = torch.zeros((grid_num, grid_num, 30))
        cell_size = 1. / grid_num
        wh = boxes[:, 2:] - boxes[:, :2]
        cxcy = (boxes[:, 2:] + boxes[:, :2]) / 2
        for i in range(cxcy.size()[0]):
            cxcy_sample = cxcy[i]
            #grid cell의 Y축과 X축의 index 계산
            ij = (cxcy_sample / cell_size).ceil() - 1
            
            # 인덱스 에러 방지 (Clamp)
            ij = ij.clamp(min=0, max=grid_num-1)
            
            #grid cell의 2개 bbox의  confidence score을 1로 set
            target[int(ij[1]), int(ij[0]), 4] = 1
            target[int(ij[1]), int(ij[0]), 9] = 1
            #grid cell의 class probability을 1로 set
            target[int(ij[1]), int(ij[0]), int(labels[i]) + 9] = 1
            #bbox의 중심점 (cx,cy)를 (i,j) grid cell의 원점으로 부터
            # offset값으로 (delta_x, delta_y) 계산하고 target 행렬 tensor의 
            # (i,j) grid cell 위치에 정규화한 bbox정보를 저장
            xy = ij * cell_size
            delta_xy = (cxcy_sample - xy) / cell_size
            target[int(ij[1]), int(ij[0]), 2:4] = wh[i]
            target[int(ij[1]), int(ij[0]), :2] = delta_xy
            target[int(ij[1]), int(ij[0]), 7:9] = wh[i]
            target[int(ij[1]), int(ij[0]), 5:7] = delta_xy
        return target

    def BGR2RGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def BGR2HSV(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def HSV2BGR(self, img):
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

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

    def randomBlur(self, bgr):
        if random.random() < 0.5:
            bgr = cv2.blur(bgr, (5, 5))
        return bgr

    def randomShift(self, bgr, boxes, labels):
        center = (boxes[:, 2:] + boxes[:, :2]) / 2
        if random.random() < 0.5:
            height, width, c = bgr.shape
            after_shfit_image = np.zeros((height, width, c), dtype=bgr.dtype)
            after_shfit_image[:, :, :] = (104, 117, 123)  # bgr
            shift_x = random.uniform(-width * 0.2, width * 0.2)
            shift_y = random.uniform(-height * 0.2, height * 0.2)

            if shift_x >= 0 and shift_y >= 0:
                after_shfit_image[int(shift_y):, int(shift_x):, :] = bgr[:height - int(shift_y), :width - int(shift_x),
                                                                     :]
            elif shift_x >= 0 and shift_y < 0:
                after_shfit_image[:height + int(shift_y), int(shift_x):, :] = bgr[-int(shift_y):, :width - int(shift_x),
                                                                              :]
            elif shift_x < 0 and shift_y >= 0:
                after_shfit_image[int(shift_y):, :width + int(shift_x), :] = bgr[:height - int(shift_y), -int(shift_x):,
                                                                             :]
            elif shift_x < 0 and shift_y < 0:
                after_shfit_image[:height + int(shift_y), :width + int(shift_x), :] = bgr[-int(shift_y):,
                                                                                      -int(shift_x):, :]

            shift_xy = torch.FloatTensor([[int(shift_x), int(shift_y)]]).expand_as(center)
            center = center + shift_xy
            mask1 = (center[:, 0] > 0) & (center[:, 0] < width)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < height)
            mask = (mask1 & mask2).view(-1, 1)
            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
            if len(boxes_in) == 0:
                return bgr, boxes, labels
            box_shift = torch.FloatTensor([[int(shift_x), int(shift_y), int(shift_x), int(shift_y)]]).expand_as(
                boxes_in)
            boxes_in = boxes_in + box_shift
            labels_in = labels[mask.view(-1)]
            return after_shfit_image, boxes_in, labels_in
        return bgr, boxes, labels

    def randomScale(self, bgr, boxes):
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            height, width, c = bgr.shape
            bgr = cv2.resize(bgr, (int(width * scale), height))
            scale_tensor = torch.FloatTensor([[scale, 1, scale, 1]]).expand_as(boxes)
            boxes = boxes * scale_tensor
            return bgr, boxes
        return bgr, boxes

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

    def random_bright(self, im, delta=16):
        alpha = random.random()
        if alpha > 0.3:
            im = im * alpha + random.randrange(-delta, delta)
            im = im.clip(min=0, max=255).astype(np.uint8)
        return im