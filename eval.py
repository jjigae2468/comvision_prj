import torch
import torch.nn as nn
import numpy as np
import cv2
from utils.util import predict 
from collections import defaultdict
from tqdm import tqdm
import time

VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

class Evaluation:
    def __init__(self, predictions, targets, threshold):
        super(Evaluation, self).__init__()
        self.predictions = predictions
        self.targets = targets
        self.threshold = threshold

    @staticmethod
    def compute_ap(recall, precision):
        recall = np.concatenate(([0.], recall, [1.]))
        precision = np.concatenate(([0.], precision, [0.]))

        for i in range(precision.size - 1, 0, -1):
            precision[i - 1] = max(precision[i - 1], precision[i])

        ap = 0.0
        for i in range(precision.size - 1):
            ap += (recall[i + 1] - recall[i]) * precision[i + 1]
        return ap

    def evaluate(self, verbose=True):
        aps = []
        if verbose:
            print(f'Evaluating with IoU Threshold {self.threshold}')
            print('CLASS'.ljust(25, ' '), 'AP')
        
        for class_name in VOC_CLASSES:
            class_preds = self.predictions[class_name]
            if len(class_preds) == 0:
                ap = 0.0
                if verbose: print(f'{class_name}'.ljust(25, ' '), '0.00 (No Preds)')
                aps.append(ap)
                continue
            
            image_ids = [x[0] for x in class_preds]
            confidence = np.array([float(x[1]) for x in class_preds])
            BB = np.array([x[2:] for x in class_preds])
            
            sorted_ind = np.argsort(-confidence)
            BB = BB[sorted_ind, :]
            image_ids = [image_ids[x] for x in sorted_ind]

            import copy
            class_targets = copy.deepcopy(self.targets)
            
            npos = 0.
            for (key1, key2) in class_targets:
                if key2 == class_name:
                    npos += len(class_targets[(key1, key2)])
            
            nd = len(image_ids)
            tp = np.zeros(nd)
            fp = np.zeros(nd)

            for d, image_id in enumerate(image_ids):
                bb = BB[d]
                if (image_id, class_name) in class_targets:
                    BBGT = class_targets[(image_id, class_name)]
                    for x1y1_x2y2 in BBGT:
                        x_min = np.maximum(x1y1_x2y2[0], bb[0])
                        y_min = np.maximum(x1y1_x2y2[1], bb[1])
                        x_max = np.minimum(x1y1_x2y2[2], bb[2])
                        y_max = np.minimum(x1y1_x2y2[3], bb[3])
                        w = np.maximum(x_max - x_min + 1., 0.)
                        h = np.maximum(y_max - y_min + 1., 0.)
                        intersection = w * h

                        union = (bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) + \
                                (x1y1_x2y2[2] - x1y1_x2y2[0] + 1.) * (x1y1_x2y2[3] - x1y1_x2y2[1] + 1.) - intersection
                        
                        if union == 0: continue

                        overlaps = intersection / union
                        if overlaps > self.threshold:
                            tp[d] = 1
                            BBGT.remove(x1y1_x2y2)
                            if len(BBGT) == 0:
                                del class_targets[(image_id, class_name)]
                            break
                    fp[d] = 1 - tp[d]
                else:
                    fp[d] = 1
            
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            recall = tp / float(npos) if npos > 0 else np.zeros(nd)
            precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

            ap = self.compute_ap(recall, precision)
            if verbose: print(f'{class_name}'.ljust(25, ' '), f'{ap:.2f}')
            aps.append(ap)

        return aps

def run_evaluation(model, device, root_path='./Dataset', batch_size=8, threshold=0.5):
    targets = defaultdict(list)
    predictions = defaultdict(list)
    image_list = []

    try:
        with open(f'{root_path}/test.txt') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Could not find {root_path}/test.txt")
        return [0.0]

    for line in lines:
        line = line.strip()
        image_name = f'{line}.jpg'
        image_list.append(image_name)

        try:
            with open(f'{root_path}/Labels/{line}.txt') as f:
                objects = f.readlines()
        except FileNotFoundError:
            continue

        for object in objects:
            c, x1, y1, x2, y2 = map(int, object.rstrip().split())
            class_name = VOC_CLASSES[c]
            targets[(image_name, class_name)].append([x1, y1, x2, y2])
            
    model.eval()
    
    total_time = 0.0
    num_images = 0
    
    with torch.no_grad():
        for image_name in tqdm(image_list, desc="Inference", leave=False):
            t0 = time.time()
            result = predict(model, image_name, root_path=f'{root_path}/Images/')
            t1 = time.time()
            
            total_time += (t1 - t0)
            num_images += 1

            for (x1, y1), (x2, y2), class_name, image_name_out, conf in result:
                predictions[class_name].append([image_name_out, conf, x1, y1, x2, y2])

    if num_images > 0:
        avg_time_ms = (total_time / num_images) * 1000
        fps = 1.0 / (total_time / num_images)
        print(f'\nInference Speed: {avg_time_ms:.2f} ms/img | FPS: {fps:.2f}')

    print("\n--- Evaluation @ IoU 0.5 ---")
    aps_05 = Evaluation(predictions, targets, threshold=0.5).evaluate(verbose=True)
    
    print("\n--- Evaluation @ IoU 0.75 ---")
    aps_075 = Evaluation(predictions, targets, threshold=0.75).evaluate(verbose=True)
    
    mean_ap_05 = np.mean(aps_05)
    mean_ap_075 = np.mean(aps_075)
    
    print(f'\nFinal mAP@0.5: {mean_ap_05:.2f}')
    print(f'Final mAP@0.75: {mean_ap_075:.2f}')
    
    return aps_05

if __name__ == '__main__':
    from nets.nn import resnet50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet50().to(device)
    run_evaluation(model, device)