import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset_loader import AlbumFeatureDataset
from models.peta import PETAModel
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
    """Lấy danh sách các lớp sự kiện từ file dataset.txt"""
    classes = set()
    with open(dataset_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '/' in line:
                classes.add(line.strip().split('/')[0])
                
    class_list = sorted(list(classes))
    return {cls_name: idx for idx, cls_name in enumerate(class_list)}

def load_pec_split(split_txt_path, class_to_idx):
    """Đọc file test.txt và map với thư mục đặc trưng"""
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
    # 1. Cấu hình đường dẫn
    DATASET_TXT = "./data/dataset.txt"
    TEST_TXT = "./data/test.txt"
    FEATURE_DIR = "./data/features"
    MODEL_WEIGHTS = "../Release/best_peta_model.pth"  # Đường dẫn tới file trọng số tốt nhất
    
    # Siêu tham số
    BATCH_SIZE = 16
    NUM_SAMPLES = 50 

    # 2. Chuẩn bị dữ liệu Test
    print("Đang chuẩn bị dữ liệu Test...")
    class_to_idx = get_class_mapping(DATASET_TXT)
    NUM_CLASSES = len(class_to_idx)
    
    test_labels_dict = load_pec_split(TEST_TXT, class_to_idx)
    
    # Khởi tạo Dataset và DataLoader cho tập Test
    test_dataset = AlbumFeatureDataset(FEATURE_DIR, test_labels_dict)
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=fixed_sample_collate)
    
    print(f"Đã nạp {len(test_dataset)} albums từ tập Test (Dữ liệu hoàn toàn mới).")
    
    # 3. Khởi tạo mô hình & Tải trọng số (Load Weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    model = PETAModel(embed_dim=2048, num_classes=NUM_CLASSES, num_heads=8, num_layers=2, dropout=0.5, max_len=NUM_SAMPLES)
    
    try:
        # Nạp trọng số đã học vào mô hình
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
        print("-> Đã tải thành công file trọng số best_peta_model.pth!")
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file trọng số tại {MODEL_WEIGHTS}. Bạn đã chạy train.py chưa?")
        return
    except RuntimeError as e:
        print("\nLỖI LỆCH TRỌNG SỐ (SHAPE MISMATCH): Cấu hình PETAModel ở file evaluate.py không khớp với lúc train.")
        print("Chi tiết lỗi:", e)
        return

    model.to(device)
    
    model.eval()
    
    # 4. Vòng lặp dự đoán trên tập Test
    test_preds = []
    test_targets = []
    
    # Tắt tính gradient để tăng tốc độ và tiết kiệm bộ nhớ
    with torch.no_grad():
        for features, labels, masks in tqdm(test_loader, desc="Đang đánh giá mô hình"):
            features, masks = features.to(device), masks.to(device)
            
            # Đưa dữ liệu qua mô hình
            outputs = model(features, masks)
            
            # Thu thập kết quả
            test_preds.append(outputs.cpu())
            test_targets.append(labels)
            
    # 5. Tính toán Độ đo tổng thể
    test_preds = torch.cat(test_preds, dim=0)
    test_targets = torch.cat(test_targets, dim=0)
    
    test_acc = calculate_accuracy(test_preds, test_targets)
    test_map = calculate_map(test_preds, test_targets, NUM_CLASSES)
    
    # 6. In Báo cáo kết quả
    print("\n" + "="*50)
    print(" BÁO CÁO KẾT QUẢ TRÊN TẬP TEST  ")
    print("="*50)
    print(f"  Accuracy : {test_acc:.4f}")
    print(f"  mAP : {test_map:.4f}")
    print("="*50)

if __name__ == "__main__":
    evaluate_model()