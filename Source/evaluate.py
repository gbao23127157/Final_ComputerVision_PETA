import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset_loader import AlbumFeatureDataset
from models.peta import PETAModel
from utils.metrics import calculate_accuracy, calculate_map

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

def evaluate_model(num_runs=5):
    DATASET_TXT = "./data/dataset.txt"
    TEST_TXT = "./data/test.txt"
    FEATURE_DIR = "./data/features"
    MODEL_WEIGHTS = "../Release/best_peta_model.pth" 
    
    BATCH_SIZE = 16
    NUM_SAMPLES = 50 

    print("Đang chuẩn bị dữ liệu Test...")
    class_to_idx = get_class_mapping(DATASET_TXT)
    NUM_CLASSES = len(class_to_idx)
    test_labels_dict = load_pec_split(TEST_TXT, class_to_idx)
    test_dataset = AlbumFeatureDataset(FEATURE_DIR, test_labels_dict)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=fixed_sample_collate)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PETAModel(embed_dim=2048, num_classes=NUM_CLASSES, num_heads=8, num_layers=2, dropout=0.5, max_len=NUM_SAMPLES)
    
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
        print("-> Đã tải thành công file trọng số best_peta_model.pth!")
    except Exception as e:
        print(f"LỖI TẢI TRỌNG SỐ: {e}")
        return

    model.to(device)
    model.eval()
    
    acc_list = []
    map_list = []

    print(f"\nBẮT ĐẦU ĐÁNH GIÁ {num_runs} LẦN \n")
    
    for run in range(num_runs):
        test_preds = []
        test_targets = []
        
        with torch.no_grad():
            for features, labels, masks in tqdm(test_loader, desc=f"Lần chạy {run+1}/{num_runs}", leave=False):
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
        print(f"   Hoàn thành Lần {run+1} | Accuracy: {test_acc:.2f}% | mAP: {test_map:.2f}%")

    # Tính Mean và Std
    mean_acc, std_acc = np.mean(acc_list), np.std(acc_list)
    mean_map, std_map = np.mean(map_list), np.std(map_list)

    print("\n" + "="*55)
    print(f" BÁO CÁO KẾT QUẢ PETA (TRUNG BÌNH SAU {num_runs} LẦN)")
    print("="*55)
    print(f"  Accuracy : {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"  mAP      : {mean_map:.2f}% ± {std_map:.2f}%")
    print("="*55)

if __name__ == "__main__":
    evaluate_model(num_runs=5)