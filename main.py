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
from torch.optim.lr_scheduler import CosineAnnealingLR # [추가]

# eval.py에서 검증 함수 가져오기
try:
    from eval import run_evaluation
except ImportError:
    print("Warning: eval.py not found or run_evaluation not implemented. mAP will not be calculated.")
    run_evaluation = None

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    root = args.data_dir
    num_epochs = args.epoch
    batch_size = args.batch_size
    learning_rate = args.lr
    
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    # [중요] nets/nn.py가 Skip-SE 구조인지 확인할 것
    net = resnet50()

    # Pretrained Weights 로드 부분
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
        
        print(f"Loading weights from {args.pre_weights}...")
        net.load_state_dict(torch.load(f'./weights/{args.pre_weights}')['state_dict'])
    else:
        epoch_start = 1
        print("Loading ImageNet pretrained ResNet50...")
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

    # 학습률 설정 (Backbone은 10배, 나머지는 1배)
    params = []
    params_dict = dict(net.named_parameters())
    for key, value in params_dict.items():
        if key.startswith('features'):
            params += [{'params': [value], 'lr': learning_rate * 10}]
        else:
            params += [{'params': [value], 'lr': learning_rate}]

    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    
    # [수정] Cosine Annealing Scheduler 적용
    # T_max는 전체 epoch 수와 맞춰야 함. 최소 LR은 1e-6까지 떨어짐.
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # 데이터셋 로드
    with open(os.path.join(root, 'train.txt')) as f:
        train_names = f.readlines()
    train_dataset = Dataset(root, train_names, train=True, transform=[transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())

    with open(os.path.join(root, 'test.txt')) as f:
        test_names = f.readlines()
    test_dataset = Dataset(root, test_names, train=False, transform=[transforms.ToTensor()])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size // 2, shuffle=False, num_workers=os.cpu_count())

    print(f'NUMBER OF DATA SAMPLES: {len(train_dataset)}')
    print(f'BATCH SIZE: {batch_size}')

    # --- 설정: mAP 검증 주기 (시간 단축용) ---
    val_interval = 5 
    
    # 로깅 및 Early Stopping 변수 초기화
    history = {
        'epoch': [], 'train_loss': [], 'val_loss': [], 'val_map_05': [], 
        'loss_xy': [], 'loss_wh': [], 'loss_obj': [], 'loss_noobj': [], 'loss_class': []
    }
    best_val_loss = float('inf')
    patience = 5 # [수정] Cosine Annealing을 쓸 때는 Patience를 조금 늘려주는 게 좋음
    patience_counter = 0
    train_start_time = time.time()

    # 결과 저장 폴더 생성
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # TXT 로그 파일 헤더 작성
    log_file_path = 'train_log.txt'
    if not os.path.exists(log_file_path) or epoch_start == 1:
        with open(log_file_path, 'w') as f:
            f.write("Epoch\tTrain Loss\tVal Loss\tVal mAP@0.5\tXY Loss\tWH Loss\tObj Loss\tNoObj Loss\tClass Loss\tTime\tLR\n")

    # --- Training Loop ---
    for epoch in range(epoch_start, num_epochs + 1):
        net.train()
        epoch_start_time = time.time()

        # [삭제] 기존 StepLR 코드 삭제됨
        
        total_loss = 0.
        epoch_loss_dict = {'loss_xy': 0, 'loss_wh': 0, 'loss_obj': 0, 'loss_noobj': 0, 'loss_class': 0}
        
        # 현재 LR 가져오기
        current_lr = optimizer.param_groups[0]['lr']
        print(('\n' + '%10s' * 4) % ('epoch', 'loss', 'gpu', 'lr'))
        progress_bar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
        
        for i, (images, target) in progress_bar:
            images = images.to(device)
            target = target.to(device)

            pred = net(images)
            
            # Loss 계산
            loss, loss_dict = criterion(pred, target.float())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            for k in epoch_loss_dict:
                epoch_loss_dict[k] += loss_dict[k]

            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
            s = ('%10s' + '%10.4g' + '%10s' + '%10.6f') % ('%g/%g' % (epoch, num_epochs), total_loss / (i + 1), mem, current_lr)
            progress_bar.set_description(s)
        
        # [추가] Epoch 끝날 때마다 Scheduler Step
        scheduler.step()

        avg_train_loss = total_loss / len(train_loader)
        
        # --- Validation Loss ---
        validation_loss = 0.0
        net.eval()
        with torch.no_grad():
            for i, (images, target) in enumerate(test_loader):
                images = images.to(device)
                target = target.to(device)
                prediction = net(images)
                loss, _ = criterion(prediction, target)
                validation_loss += loss.item()
        validation_loss /= len(test_loader)
        
        # --- mAP Calculation ---
        val_map = 0.0
        if len(history['val_map_05']) > 0:
            val_map = history['val_map_05'][-1]
            
        do_full_eval = (epoch % val_interval == 0) or (epoch == 1)

        if do_full_eval and run_evaluation:
            print(f"\n[FULL EVAL] Evaluating mAP for Epoch {epoch}...")
            aps = run_evaluation(net, device, root_path=args.data_dir, batch_size=batch_size, threshold=0.5)
            val_map = np.mean(aps)
        
        # --- ETA 및 통계 출력 ---
        epoch_duration = time.time() - epoch_start_time
        elapsed_time = time.time() - train_start_time
        remaining_epochs = num_epochs - epoch
        eta = remaining_epochs * epoch_duration
        eta_str = str(datetime.timedelta(seconds=int(eta)))
        
        avg_xy = epoch_loss_dict['loss_xy'] / len(train_loader)
        avg_wh = epoch_loss_dict['loss_wh'] / len(train_loader)
        avg_obj = epoch_loss_dict['loss_obj'] / len(train_loader)
        avg_noobj = epoch_loss_dict['loss_noobj'] / len(train_loader)
        avg_class = epoch_loss_dict['loss_class'] / len(train_loader)

        print(f'\nEpoch [{epoch}/{num_epochs}] Train Loss: {avg_train_loss:.4f}, Val Loss: {validation_loss:.4f}, Val mAP: {val_map:.4f}')
        print(f'   Details -> XY: {avg_xy:.4f} | WH: {avg_wh:.4f} | Obj: {avg_obj:.4f} | NoObj: {avg_noobj:.4f} | Class: {avg_class:.4f}')
        print(f'ETA: {eta_str} (Elapsed: {str(datetime.timedelta(seconds=int(elapsed_time)))})')

        # --- 로그 저장 ---
        history['epoch'].append(epoch)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(validation_loss)
        history['val_map_05'].append(val_map)
        for k in epoch_loss_dict:
            history[k].append(epoch_loss_dict[k] / len(train_loader))

        with open(log_file_path, 'a') as f:
            f.write(f"{epoch}\t{avg_train_loss:.4f}\t{validation_loss:.4f}\t{val_map:.4f}\t"
                    f"{avg_xy:.4f}\t{avg_wh:.4f}\t{avg_obj:.4f}\t{avg_noobj:.4f}\t{avg_class:.4f}\t"
                    f"{epoch_duration:.2f}\t{current_lr:.6f}\n")

        # --- Early Stopping & Best Model Save ---
        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            patience_counter = 0
            torch.save({'state_dict': net.state_dict()}, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"Saved Best Model at Epoch {epoch} (Val Loss: {validation_loss:.4f})")
        else:
            patience_counter += 1
            print(f"Early Stopping Counter: {patience_counter}/{patience}")
            # Cosine Annealing 중에는 Loss가 일시적으로 튈 수 있으니 Early Stopping을 덜 민감하게 하는 게 좋음
            if patience_counter >= patience:
                print("Early Stopping Triggered!")
                break

        if (epoch % 10) == 0:
            save = {'state_dict': net.state_dict()}
            torch.save(save, os.path.join(args.save_dir, f'yolov1_{epoch:04d}.pth'))

    # --- 학습 종료 후 그래프 저장 ---
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
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--data_dir", type=str, default='./Dataset')
    parser.add_argument("--pre_weights", type=str, help="pretrained weight")
    parser.add_argument("--save_dir", type=str, default="./weights")
    parser.add_argument("--img_size", type=int, default=448)
    args = parser.parse_args()
    
    main(args)