import os
import tqdm
import numpy as np
import torch
import torchvision
from torchvision import transforms
from nets.nn import resnet50 
from utils.loss import yoloLoss
from utils.dataset import Dataset
import argparse
import re
import time
import matplotlib.pyplot as plt
import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from eval import run_evaluation
except ImportError:
    run_evaluation = None

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    root = args.data_dir
    num_epochs = args.epoch
    batch_size = args.batch_size
    learning_rate = args.lr
    eval_interval = 5
    
    # [New] Warmup 설정
    warmup_epochs = 5
    
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    net = resnet50() 
    
    if(args.pre_weights != None):
        pattern = 'yolov1_([0-9]+)'
        strs = args.pre_weights.split('.')[-2]
        f_name = strs.split('/')[-1]
        match = re.search(pattern, f_name)
        if match:
            epoch_str = match.group(1)
            epoch_start = int(epoch_str) + 1
        else:
            epoch_start = 1
        
        print(f'Loading weights from {args.pre_weights}...')
        state_dict = torch.load(args.pre_weights, map_location=device)['state_dict']
        net.load_state_dict(state_dict)
    else:
        epoch_start = 1
        print('Training from scratch...')

    net = net.to(device)
    net.train()

    params = []
    params_dict = dict(net.named_parameters())
    for key, value in params_dict.items():
        if key.startswith('features'):
            params += [{'params': [value], 'lr': learning_rate * 1}]
        else:
            params += [{'params': [value], 'lr': learning_rate}]
    
    optimizer = torch.optim.SGD(
        params,
        lr=learning_rate,
        momentum=0.9,
        weight_decay=5e-4
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.00001)
    criterion = yoloLoss()

    try:
        train_txt_path = os.path.join(root, 'train.txt')
        val_txt_path = os.path.join(root, 'test.txt') 

        train_dataset = Dataset(root, open(train_txt_path).readlines(), train=True, transform=[transforms.ToTensor()])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        val_dataset = Dataset(root, open(val_txt_path).readlines(), train=False, transform=[transforms.ToTensor()])
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    except FileNotFoundError as e:
        print(f"\n[Error] 파일을 찾을 수 없습니다: {e}")
        return

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    log_file_path = 'train_log.txt' 
    if not os.path.exists(log_file_path):
        with open(log_file_path, 'w') as f:
            f.write("Epoch\tTrain Loss\tVal Loss\tVal mAP@0.5\tXY Loss\tWH Loss\tObj Loss\tNoObj Loss\tClass Loss\tTime\tLR\n")

    print(f'Training starts from epoch {epoch_start} to {num_epochs}')
    print(f'Total batches per epoch: {len(train_loader)}')

    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_map_05': []}
    train_start_time = time.time()
    best_map = 0.0
    current_map = 0.0

    for epoch in range(epoch_start, num_epochs + 1):
        net.train()
        epoch_start_time = time.time()
        
        # [Mosaic/Mixup 제어] 마지막 10 Epoch는 증강 끄고 Fine-tuning
        if num_epochs - epoch < 10:
            if hasattr(train_dataset, 'enable_mosaic') and train_dataset.enable_mosaic:
                print(f"==> [Epoch {epoch}] Augmentation Disabled for Fine-tuning! <==")
                train_dataset.enable_mosaic = False
                train_dataset.enable_mixup = False # Mixup도 끔

        running_losses = {
            'total': 0., 'xy': 0., 'wh': 0., 'obj': 0., 'noobj': 0., 'class': 0.
        }
        
        pbar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}')
        
        for i, (images, target) in enumerate(pbar):
            # [New] Warmup Logic
            # 학습 초반 (5 에폭 미만) 동안 LR을 선형적으로 증가
            if epoch <= warmup_epochs:
                current_iter = (epoch - 1) * len(train_loader) + i
                warmup_total_iters = warmup_epochs * len(train_loader)
                warmup_lr = learning_rate * (current_iter / warmup_total_iters)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            
            images, target = images.to(device), target.to(device)
            pred = net(images)
            
            loss, loss_dict = criterion(pred, target)
            
            running_losses['total'] += loss.item()
            running_losses['xy'] += loss_dict['loss_xy']
            running_losses['wh'] += loss_dict['loss_wh']
            running_losses['obj'] += loss_dict['loss_obj']
            running_losses['noobj'] += loss_dict['loss_noobj']
            running_losses['class'] += loss_dict['loss_class']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({'Loss': loss.item(), 'LR': optimizer.param_groups[0]['lr']})
        
        # 평균 Loss 계산
        num_batches = len(train_loader)
        avg_losses = {k: v / num_batches for k, v in running_losses.items()}
        
        # Scheduler Step (Warmup 이후부터 정상 스케줄링)
        if epoch > warmup_epochs:
            scheduler.step()
        
        net.eval()
        val_loss = 0.
        with torch.no_grad():
             for images, target in val_loader:
                 images, target = images.to(device), target.to(device)
                 pred = net(images)
                 loss, _ = criterion(pred, target)
                 val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        
        if run_evaluation and (epoch % eval_interval == 0 or epoch == num_epochs):
            print(f"Evaluating Epoch {epoch}...")
            try:
                aps = run_evaluation(net, device, root_path=root, batch_size=batch_size, threshold=0.5)
                current_map = np.mean(aps)
            except Exception as e:
                print(f"Evaluation failed: {e}")
        
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']

        log_line = f"{epoch}\t{avg_losses['total']:.4f}\t{avg_val_loss:.4f}\t{current_map:.4f}\t" \
                   f"{avg_losses['xy']:.4f}\t{avg_losses['wh']:.4f}\t{avg_losses['obj']:.4f}\t" \
                   f"{avg_losses['noobj']:.4f}\t{avg_losses['class']:.4f}\t{epoch_time:.2f}\t{current_lr:.6f}\n"
        
        with open(log_file_path, 'a') as f:
            f.write(log_line)

        print(f'\nEpoch [{epoch}/{num_epochs}] Time: {epoch_time:.2f}s | Train Loss: {avg_losses["total"]:.4f} | Val mAP: {current_map:.4f}')

        history['epoch'].append(epoch)
        history['train_loss'].append(avg_losses['total'])
        history['val_loss'].append(avg_val_loss)
        history['val_map_05'].append(current_map)

        if current_map > best_map:
            best_map = current_map
            torch.save({'state_dict': net.state_dict()}, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"Best model saved with mAP: {best_map:.4f}")
            
        if epoch % 10 == 0:
            torch.save({'state_dict': net.state_dict()}, os.path.join(args.save_dir, f'yolov1_{epoch:04d}.pth'))

    print("Training Finished. Saving graphs...")
    plt.figure(figsize=(10, 5))
    plt.plot(history['epoch'], history['train_loss'], label='Train Loss')
    plt.plot(history['epoch'], history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('loss_curve.png')
    
    plt.figure(figsize=(10, 5))
    plt.plot(history['epoch'], history['val_map_05'], label='Val mAP@0.5', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.legend()
    plt.title('Validation mAP')
    plt.savefig('map_curve.png')

    print(f"Total Training Time: {str(datetime.timedelta(seconds=int(time.time() - train_start_time)))}")
    save = {'state_dict': net.state_dict()}
    torch.save(save, os.path.join(args.save_dir, 'yolov1_final.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./Dataset")
    parser.add_argument("--save_dir", type=str, default="./weights")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epoch", type=int, default=40) 
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--pre_weights", type=str, default=None)
    
    args = parser.parse_args()
    main(args)