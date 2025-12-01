import os
import tqdm
import numpy as np
import time
from datetime import datetime, timedelta
from torchsummary import summary

import torch
import torchvision
from torchvision import transforms

from nets.nn import resnet50
from utils.loss import yoloLoss
from utils.dataset import Dataset

import argparse
import re

# mAP 계산을 위한 import
from eval import Evaluation
from utils.util import predict, VOC_CLASSES
from collections import defaultdict

# Early Stopping 클래스
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='./weights/best_model.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        # [수정된 부분]: np.Inf -> np.inf (NumPy 2.0 에러 해결)
        self.val_loss_min = np.inf 
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        if isinstance(model, torch.nn.DataParallel):
            model_to_save = model.module
        else:
            model_to_save = model
            
        save = {'state_dict': model_to_save.state_dict()}
        torch.save(save, self.path)
        self.val_loss_min = val_loss

# mAP 계산 함수
def compute_mAP(model, root_path='./Dataset'):
    model.eval()
    targets = defaultdict(list)
    predictions = defaultdict(list)
    image_list = []
    
    with open(f'{root_path}/test.txt') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        image_name = f'{line}.jpg'
        image_list.append(image_name)
        
        with open(f'{root_path}/Labels/{line}.txt') as f:
            objects = f.readlines()
        for object in objects:
            c, x1, y1, x2, y2 = map(int, object.rstrip().split())
            class_name = VOC_CLASSES[c]
            targets[(image_name, class_name)].append([x1, y1, x2, y2])
            
    print("Calculating mAP... (This may take a while)")
    with torch.no_grad():
        for image_name in tqdm.tqdm(image_list, desc="Evaluating"): 
             result = predict(model, image_name, root_path=f'{root_path}/Images/')
             for (x1, y1), (x2, y2), class_name, img_name, conf in result:
                predictions[class_name].append([img_name, conf, x1, y1, x2, y2])
                
    aps = Evaluation(predictions, targets, threshold=0.5).evaluate()
    mAP = np.mean(aps)
    return mAP

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    root = args.data_dir
    num_epochs = args.epoch
    batch_size = args.batch_size
    learning_rate = args.lr
    
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    net = resnet50()

    if(args.pre_weights != None):
        pattern = 'yolov1_([0-9]+)'
        strs = args.pre_weights.split('.')[-2]
        f_name = strs.split('/')[-1]
        epoch_str = re.search(pattern,f_name).group(1)
        epoch_start = int(epoch_str) + 1
        net.load_state_dict( \
            torch.load(f'./weights/{args.pre_weights}')['state_dict'])
    else:
        epoch_start = 1
        # weights 인자 사용 권장 (UserWarning 대응)
        try:
            from torchvision.models import ResNet50_Weights
            resnet = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
        except ImportError:
            resnet = torchvision.models.resnet50(pretrained=True)
            
        new_state_dict = resnet.state_dict()
    
        net_dict = net.state_dict()
        for k in new_state_dict.keys():
            if k in net_dict.keys() and not k.startswith('fc'):
                net_dict[k] = new_state_dict[k]
        net.load_state_dict(net_dict)

    print('NUMBER OF CUDA DEVICES:', torch.cuda.device_count())

    criterion = yoloLoss().to(device)
    net = net.to(device)

    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    net.train()

    params = []
    params_dict = dict(net.named_parameters())
    for key, value in params_dict.items():
        if key.startswith('features'):
            params += [{'params': [value], 'lr': learning_rate * 10}]
        else:
            params += [{'params': [value], 'lr': learning_rate}]

    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    with open('./Dataset/train.txt') as f:
        train_names = f.readlines()
    train_dataset = Dataset(root, train_names, train=True, transform=[transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                            num_workers=os.cpu_count())

    with open('./Dataset/test.txt') as f:
        test_names = f.readlines()
    test_dataset = Dataset(root, test_names, train=False, transform=[transforms.ToTensor()])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size // 2, shuffle=False,
                                            num_workers=os.cpu_count())

    print(f'NUMBER OF DATA SAMPLES: {len(train_dataset)}')
    print(f'BATCH SIZE: {batch_size}')

    # 로그 파일 초기화 (ETA, Epoch Time 추가)
    with open('train_log.txt', 'w') as f:
        f.write('Epoch,LR,Total_Loss,Coord_Loss,Obj_Loss,NoObj_Loss,Class_Loss,Val_Loss,mAP,Epoch_Time(s),ETA\n')

    early_stopping = EarlyStopping(patience=3, verbose=True, path='./weights/best_model.pth')
    current_map = 0.0
    start_train_time = time.time() 

    for epoch in range(epoch_start, num_epochs + 1):
        epoch_start_time = time.time() # 에폭 시작 시간 기록
        
        net.train()

        if epoch == 30:
            learning_rate = 0.0001
        if epoch == 40:
            learning_rate = 0.00001
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        # Training
        total_loss = 0.
        total_xy = 0.
        total_wh = 0.
        total_obj = 0.
        total_noobj = 0.
        total_cls = 0.

        print(('\n' + '%10s' * 3) % ('epoch', 'loss', 'gpu'))
        progress_bar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (images, target) in progress_bar:
            images = images.to(device)
            target = target.to(device)

            pred = net(images)
            optimizer.zero_grad()
            
            # Loss 계산 및 Unpacking (loss.py의 6개 값 리턴)
            loss, xy_l, wh_l, obj_l, noobj_l, cls_l = criterion(pred, target.float())

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_xy += xy_l.item()
            total_wh += wh_l.item()
            total_obj += obj_l.item()
            total_noobj += noobj_l.item()
            total_cls += cls_l.item()

            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
            s = ('%10s' + '%10.4g' + '%10s') % ('%g/%g' % (epoch, num_epochs), total_loss / (i + 1), mem)
            progress_bar.set_description(s)
        
        # 평균 Loss 계산
        avg_loss = total_loss / len(train_loader)
        avg_coord = (total_xy + total_wh) / len(train_loader)
        avg_obj = total_obj / len(train_loader)
        avg_noobj = total_noobj / len(train_loader)
        avg_cls = total_cls / len(train_loader)

        # Validation
        validation_loss = 0.0
        net.eval()
        with torch.no_grad():
            for i, (images, target) in enumerate(test_loader):
                images = images.to(device)
                target = target.to(device)
                prediction = net(images)
                
                val_res = criterion(prediction, target)
                if isinstance(val_res, tuple):
                    loss = val_res[0]
                else:
                    loss = val_res
                validation_loss += loss.item()
            
        avg_val_loss = validation_loss / len(test_loader)
        print(f'Validation_Loss: {avg_val_loss:.4f}')
        
        # mAP 계산 (5 에폭 간격)
        if epoch % 5 == 0 or epoch == num_epochs:
            print(f"\nEvaluating mAP for epoch {epoch}...")
            current_map = compute_mAP(net, root_path=root)
            print(f"Epoch {epoch} mAP: {current_map:.4f}")
            net.train()

        # 시간 계산 (에폭 소요 시간 & ETA)
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time # 현재 에폭 걸린 시간 (초)
        
        current_time = time.time()
        elapsed_total = current_time - start_train_time
        completed_epochs = epoch - epoch_start + 1
        
        avg_time_per_epoch = elapsed_total / completed_epochs # 평균 에폭 시간
        remaining_epochs = num_epochs - epoch
        remaining_time = avg_time_per_epoch * remaining_epochs
        
        finish_time = datetime.now() + timedelta(seconds=remaining_time)
        finish_time_str = finish_time.strftime('%m-%d %H:%M')
        
        print(f"Epoch Duration: {epoch_duration:.2f}s | ETA: {finish_time_str}")

        # 로그 파일 쓰기
        current_lr = optimizer.param_groups[0]['lr']
        with open('train_log.txt', 'a') as f:
            f.write(f'{epoch},{current_lr:.6f},{avg_loss:.4f},{avg_coord:.4f},{avg_obj:.4f},{avg_noobj:.4f},{avg_cls:.4f},{avg_val_loss:.4f},{current_map:.4f},{epoch_duration:.2f},{finish_time_str}\n')

        # Early Stopping
        early_stopping(avg_val_loss, net)
        if early_stopping.early_stop:
            print("Early stopping triggered! Training stopped.")
            break
            
        if (epoch % 10) == 0:
            save = {'state_dict': net.state_dict()}
            torch.save(save, f'./weights/yolov1_{epoch:04d}.pth')

    save = {'state_dict': net.state_dict()}
    torch.save(save, './weights/yolov1_final.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--data_dir", type=str, default='./Dataset')
    parser.add_argument("--pre_weights", type=str, help="pretrained weight")
    parser.add_argument("--save_dir", type=str, default="./weights")
    parser.add_argument("--img_size", type=int, default=448)
    args = parser.parse_args()
    main(args)