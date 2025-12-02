import os
import cv2
import torch
import torch.nn as nn
from nets.nn import resnet50
from utils.util import predict
import argparse
import matplotlib.pyplot as plt

VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

COLORS = {'aeroplane': (0, 0, 0),
          'bicycle': (128, 0, 0),
          'bird': (0, 128, 0),
          'boat': (128, 128, 0),
          'bottle': (0, 0, 128),
          'bus': (128, 0, 128),
          'car': (0, 128, 128),
          'cat': (128, 128, 128),
          'chair': (64, 0, 0),
          'cow': (192, 0, 0),
          'diningtable': (64, 128, 0),
          'dog': (192, 128, 0),
          'horse': (64, 0, 128),
          'motorbike': (192, 0, 128),
          'person': (64, 128, 128),
          'pottedplant': (192, 128, 128),
          'sheep': (0, 64, 0),
          'sofa': (128, 64, 0),
          'train': (0, 192, 0),
          'tvmonitor': (128, 192, 0)}

def detect(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet50().to(device)

    print('LOADING MODEL...')
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # 가중치 경로 처리
    weight_path = args.weight
    if not os.path.exists(weight_path):
        weight_path = os.path.join('./weights', args.weight)
        
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device)['state_dict'])
        print(f'Loaded weights from {weight_path}')
    else:
        print(f"Error: Weight file not found at {args.weight} or ./weights/{args.weight}")
        return

    model.eval()
    
    with torch.no_grad():
        # 이미지 경로 처리
        if os.path.exists(args.image):
            image_name = args.image
        else:
            image_name = os.path.join("./Dataset/Images/", args.image)
            
        if not os.path.exists(image_name):
            print(f"Error: Image file not found at {image_name}")
            return

        image = cv2.imread(image_name)
        if image is None:
            print(f"Error: Could not read image at {image_name}. Check file integrity.")
            return

        print(f'\nPREDICTING on {image_name}...')
        result = predict(model, image_name)

    # 결과 시각화 (박스 그리기)
    for x1y1, x2y2, class_name, _, prob in result:
        color = COLORS[class_name]
        cv2.rectangle(image, x1y1, x2y2, color, 2)

        label = class_name + str(round(prob, 2))
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)

        p1 = (x1y1[0], x1y1[1] - text_size[1])
        cv2.rectangle(image, (p1[0] - 2 // 2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]),
                      color, -1)
        cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, 8)

    # [수정] 무조건 저장 (result.jpg)
    save_path = './result.jpg'
    cv2.imwrite(save_path, image)
    print(f"\n[Done] Result saved to: {os.path.abspath(save_path)}")
    print("Please download 'result.jpg' to view the detection result.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default='000001.jpg', required=False, help='Path to Image file')
    parser.add_argument('--video', default='', required=False, help='Path to Video file')
    parser.add_argument('--weight', default='best_model.pth', required=False, help='Weight file path')
    args = parser.parse_args()

    detect(args)