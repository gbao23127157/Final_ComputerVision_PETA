import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset_loader import AlbumFeatureDataset
from models.baseline import BaselineModel
from utils.metrics import calculate_accuracy, calculate_map

# Hàm lấy mẫu cố định 
def fixed_sample_collate(batch):
    num_samples = 50 # Số lượng ảnh SA quy định
    
    features_list = []
    labels_list = []
    masks_list = []
    
    for features, label in batch:
        n = features.size(0)
        
        if n >= num_samples:
            # Lấy mẫu ngẫu nhiên không lặp lại
            indices = torch.randperm(n)[:num_samples]
            sampled_features = features[indices]
        else:
            # Lặp lại ảnh ngẫu nhiên để bù vào phần thiếu
            missing_count = num_samples - n
            duplicate_indices = torch.randint(0, n, (missing_count,))
            duplicated_features = features[duplicate_indices]
            sampled_features = torch.cat([features, duplicated_features], dim=0)
        
        features_list.append(sampled_features)
        labels_list.append(label)
        
        # Mask toàn số 1
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
    DATASET_TXT = "./data/dataset.txt"
    TEST_TXT = "./data/test.txt"
    FEATURE_DIR = "./data/features"
    MODEL_WEIGHTS = "../Release/best_baseline_model.pth" 
    
    BATCH_SIZE = 16

    print("Đang chuẩn bị dữ liệu Test...")
    class_to_idx = get_class_mapping(DATASET_TXT)
    NUM_CLASSES = len(class_to_idx)
    test_labels_dict = load_pec_split(TEST_TXT, class_to_idx)
    
    test_dataset = AlbumFeatureDataset(FEATURE_DIR, test_labels_dict)
    
    # [ĐÃ SỬA]: Sử dụng fixed_sample_collate thay vì pad_album_features
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=fixed_sample_collate)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    model = BaselineModel(embed_dim=2048, num_classes=NUM_CLASSES, dropout=0.3)
    
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
        print(f"-> Đã tải thành công file trọng số {MODEL_WEIGHTS}!")
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file trọng số tại {MODEL_WEIGHTS}.")
        return

    model.to(device)
    model.eval()
    
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for features, labels, masks in tqdm(test_loader, desc="Đang đánh giá mô hình"):
            features, masks = features.to(device), masks.to(device)
            outputs = model(features, masks)
            test_preds.append(outputs.cpu())
            test_targets.append(labels)
            
    test_preds = torch.cat(test_preds, dim=0)
    test_targets = torch.cat(test_targets, dim=0)
    
    test_acc = calculate_accuracy(test_preds, test_targets)
    test_map = calculate_map(test_preds, test_targets, NUM_CLASSES)
    
    print("\n" + "="*50)
    print(" BÁO CÁO KẾT QUẢ TRÊN TẬP TEST")
    print("="*50)
    print(f"  Accuracy : {test_acc:.4f}")
    print(f"  mAP : {test_map:.4f}")
    print("="*50)

if __name__ == "__main__":
    evaluate_model()