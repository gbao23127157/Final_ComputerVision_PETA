import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset_loader import AlbumFeatureDataset
from utils.metrics import calculate_accuracy, calculate_map

# Import 4 phiên bản mô hình
from models.baseline import BaselineModel
from models.peta import PETAModel as PETA
from models.peta_clip import PETAModel as PETAClip
from models.peta_cross import PETAModel as PETACross

def fixed_sample_collate(batch):
    num_samples = 50 
    
    features_list = []
    labels_list = []
    masks_list = []
    
    for features, label in batch:
        n = features.size(0)
        if n >= num_samples:
            indices = torch.randperm(n)[:num_samples]
            sampled_features = features[indices]
        else:
            missing_count = num_samples - n
            duplicate_indices = torch.randint(0, n, (missing_count,))
            duplicated_features = features[duplicate_indices]
            sampled_features = torch.cat([features, duplicated_features], dim=0)
        
        features_list.append(sampled_features)
        labels_list.append(label)
        masks_list.append(torch.ones(num_samples))
        
    return torch.stack(features_list), torch.tensor(labels_list), torch.stack(masks_list)

def get_class_mapping(dataset_txt_path):
    classes = set()
    with open(dataset_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '/' in line:
                classes.add(line.strip().split('/')[0])
    class_list = sorted(list(classes))
    return {cls_name: idx for idx, cls_name in enumerate(class_list)}

def load_pec_split(split_txt_path, class_to_idx):
    labels_dict = {}
    with open(split_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '/' in line:
                class_name, album_id = line.split('/')
                album_folder_name = f"{class_name}_{album_id}"
                labels_dict[album_folder_name] = class_to_idx[class_name]
    return labels_dict

def evaluate_model():
    parser = argparse.ArgumentParser(description="Đánh giá 4 phiên bản mô hình")
    parser.add_argument('--mode', type=str, default='cross', 
                        choices=['baseline', 'peta_base', 'peta_clip', 'peta_cross'],
                        help='Chọn phiên bản mô hình')
    parser.add_argument('--runs', type=int, default=5, 
                        help='Số lần đánh giá để lấy trung bình')
    args = parser.parse_args()

    DATASET_TXT = "./data/dataset.txt"
    TEST_TXT = "./data/test.txt"
    BATCH_SIZE = 16
    NUM_SAMPLES = 50 

    print("="*55)
    print(f" Test: {args.mode.upper()}")
    print("="*55)

    class_to_idx = get_class_mapping(DATASET_TXT)
    NUM_CLASSES = len(class_to_idx)

    if args.mode == 'baseline':
        FEATURE_DIR = "./data/features/resnet"
        MODEL_WEIGHTS = "./weights/baseline_weights.pth"
        model = BaselineModel(embed_dim=2048, num_classes=NUM_CLASSES)
        
    elif args.mode == 'peta_base':
        FEATURE_DIR = "./data/features/resnet"
        MODEL_WEIGHTS = "./weights/peta_base_weights.pth"
        model = PETA(embed_dim=2048, num_classes=NUM_CLASSES, num_heads=8, num_layers=2, max_len=NUM_SAMPLES)
        
    elif args.mode == 'peta_clip':
        FEATURE_DIR = "./data/features/clip"
        MODEL_WEIGHTS = "./weights/peta_clip_weights.pth"
        model = PETAClip(embed_dim=512, num_classes=NUM_CLASSES, num_heads=8, num_layers=2)
        
    elif args.mode == 'peta_cross':
        FEATURE_DIR = "./data/features/clip"
        MODEL_WEIGHTS = "./weights/peta_cross_weights.pth"
        model = PETACross(embed_dim=512, num_classes=NUM_CLASSES, num_heads=8, num_layers=2)

    print(f"-> Thư mục dữ liệu: {FEATURE_DIR}")
    
    test_labels_dict = load_pec_split(TEST_TXT, class_to_idx)
    test_dataset = AlbumFeatureDataset(FEATURE_DIR, test_labels_dict)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=fixed_sample_collate)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
        print(f"-> Đã tải thành công mô hình: {MODEL_WEIGHTS}")
    except Exception as e:
        print(f"LỖI TẢI TRỌNG SỐ: {e}")
        print(f"Hãy chạy lệnh `python train.py --mode {args.mode}` trước để tạo file trọng số.")
        return

    model.eval()
    
    acc_list = []
    map_list = []

    print(f"\nBẮT ĐẦU ĐÁNH GIÁ {args.runs} LẦN \n")
    
    for run in range(args.runs):
        test_preds = []
        test_targets = []
        
        with torch.no_grad():
            for features, labels, masks in tqdm(test_loader, desc=f"Lần chạy {run+1}/{args.runs}", leave=False):
                features, masks = features.to(device), masks.to(device)
                outputs = model(features, masks)
                test_preds.append(outputs.cpu())
                test_targets.append(labels)
                
        test_preds = torch.cat(test_preds, dim=0)
        test_targets = torch.cat(test_targets, dim=0)
        
        test_acc = calculate_accuracy(test_preds, test_targets) * 100
        test_map = calculate_map(test_preds, test_targets, NUM_CLASSES) * 100
        
        acc_list.append(test_acc)
        map_list.append(test_map)
        print(f"   Lần {run+1} | Accuracy: {test_acc:.2f}% | mAP: {test_map:.2f}%")

    mean_acc, std_acc = np.mean(acc_list), np.std(acc_list)
    mean_map, std_map = np.mean(map_list), np.std(map_list)

    print("\n" + "="*55)
    print(f" BÁO CÁO KẾT QUẢ: {args.mode.upper()} (TRUNG BÌNH SAU {args.runs} LẦN)")
    print("="*55)
    print(f"  Accuracy : {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"  mAP      : {mean_map:.2f}% ± {std_map:.2f}%")
    print("="*55)

if __name__ == "__main__":
    evaluate_model()