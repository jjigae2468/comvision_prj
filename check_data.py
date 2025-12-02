import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from utils.dataset import Dataset
import torchvision.transforms as transforms

# VOC 클래스 정의
VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

def check_class_distribution(root='./Dataset'):
    print("Checking Class Distribution...")
    counts = defaultdict(int)
    
    with open(os.path.join(root, 'train.txt')) as f:
        lines = f.readlines()
        
    for line in tqdm(lines):
        label_path = os.path.join(root, 'Labels', line.strip() + '.txt')
        if not os.path.exists(label_path): continue
        
        with open(label_path) as lf:
            objects = lf.readlines()
            for obj in objects:
                c = int(obj.split()[0])
                counts[VOC_CLASSES[c]] += 1
                
    print("\n[Class Distribution Results]")
    # 개수 순으로 정렬하여 출력
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    for cls, count in sorted_counts:
        print(f"{cls.ljust(15)}: {count}")
    
    return sorted_counts

def visualize_augmented_data(root='./Dataset'):
    print("\nVisualizing Augmented Data...")
    with open(os.path.join(root, 'train.txt')) as f:
        train_names = f.readlines()[:10] # 10장만 샘플링

    # Augmentation이 적용된 데이터셋 로드
    dataset = Dataset(root, train_names, train=True, transform=[])
    
    save_dir = 'augmented_samples'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"Saving 10 augmented samples to './{save_dir}'...")
    
    for i in range(10):
        # Dataset의 __getitem__을 호출하여 증강된 이미지와 타겟을 가져옴
        img, target = dataset[i] 
        
        # 이미지 복원 (Normalizing 해제 및 BGR 변환)
        # dataset.py에서 subMean을 하므로 다시 더해줌
        img = img + np.array((123, 117, 104), dtype=np.float32)
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Grid 정보를 기반으로 박스 복원하여 그리기 (검증용)
        # (Dataset의 encoder 로직 역산)
        grid_num = 14
        cell_size = 1.0 / grid_num
        h, w, _ = img.shape
        
        for y in range(grid_num):
            for x in range(grid_num):
                # Confidence가 1인 경우만 (즉, 물체가 있는 그리드)
                if target[y, x, 4] == 1: 
                    # 상대 좌표 복원
                    # target 구조: [x, y, w, h, conf, ...]
                    # dataset.py의 encoder를 보면 target[y,x,:2]가 delta_xy임
                    
                    # box 중심 (grid cell 기준 상대좌표 -> 전체 정규 좌표)
                    delta_xy = target[y, x, :2]
                    wh = target[y, x, 2:4]
                    
                    cx = (x + delta_xy[0]) * cell_size
                    cy = (y + delta_xy[1]) * cell_size
                    
                    # box 크기 복원
                    bw = wh[0]
                    bh = wh[1]
                    
                    # 절대 좌표 변환
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)
                    
                    # 박스 그리기
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.imwrite(f'{save_dir}/sample_{i}.jpg', img)
    
    print("Done. Check the 'augmented_samples' folder.")

if __name__ == '__main__':
    # 1. 클래스 분포 확인
    check_class_distribution()
    
    # 2. 증강된 이미지 눈으로 확인
    visualize_augmented_data()