import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# import từ các module đã build
from data.dataset_loader import AlbumFeatureDataset
from data.preprocess import pad_album_features
from models.peta import PETAModel
from utils.metrics import calculate_accuracy, calculate_map
from utils.logger import setup_logger

def get_class_mapping(dataset_txt_path):
    """
    Quét file dataset.txt để tìm tất cả các lớp sự kiện (birthday, wedding...)
    và gán cho chúng một ID số nguyên (0, 1, 2...).
    """
    classes = set()
    with open(dataset_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '/' in line:
                class_name = line.strip().split('/')[0]
                classes.add(class_name)
                
    # Sắp xếp theo alphabet để đảm bảo ID luôn cố định mỗi lần chạy
    class_list = sorted(list(classes))
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_list)}
    
    print(f"Đã ánh xạ {len(class_to_idx)} lớp sự kiện: {class_to_idx}")
    return class_to_idx

def load_pec_split(split_txt_path, class_to_idx):
    """
    Đọc file train.txt hoặc test.txt của PEC.
    Định dạng: <class_name>/<album_id> (VD: birthday/100)
    Chuyển thành dict: {'birthday_100': 0}
    """
    labels_dict = {}
    with open(split_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '/' in line:
                class_name, album_id = line.split('/')
                # Khớp với tên folder "birthday_100" ta vừa tạo ở Bước 1
                album_folder_name = f"{class_name}_{album_id}"
                
                # Gán nhãn số nguyên
                labels_dict[album_folder_name] = class_to_idx[class_name]
                
    return labels_dict

# Thêm tham số scheduler vào hàm train_model
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, num_classes, logger, save_path):
    """
    Hàm thực hiện vòng lặp huấn luyện và đánh giá mô hình qua nhiều epoch.
    Lưu lại mô hình có độ chính xác trên tập validation cao nhất.
    """
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        logger.info(f"--- Epoch {epoch + 1}/{num_epochs} ---")
        
        # 1. PHA HUẤN LUYỆN (TRAINING)
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []

        # Tích hợp thanh tiến trình cho vòng lặp train
        for features, labels, masks in tqdm(train_loader, desc="Huấn luyện", leave=False):
            # Đưa dữ liệu lên GPU/CPU
            features, labels, masks = features.to(device), labels.to(device), masks.to(device)

            # Xóa bỏ đạo hàm (gradient) cũ từ bước trước
            optimizer.zero_grad()

            # Truyền dữ liệu tiến qua mô hình (Forward pass)
            outputs = model(features, masks)
            
            # Tính toán độ lỗi (Loss)
            loss = criterion(outputs, labels)
            
            # Lan truyền ngược (Backward pass) để tính đạo hàm
            loss.backward()
            
            # Cập nhật trọng số của mạng
            optimizer.step()

            # Lưu lại thông tin để tính chỉ số mAP và Accuracy
            train_loss += loss.item() * features.size(0)
            train_preds.append(outputs.detach())
            train_targets.append(labels.detach())

        # Tính toán độ đo tổng thể cho tập Train
        train_loss = train_loss / len(train_loader.dataset)
        train_preds = torch.cat(train_preds, dim=0)
        train_targets = torch.cat(train_targets, dim=0)
        train_acc = calculate_accuracy(train_preds, train_targets)
        train_map = calculate_map(train_preds, train_targets, num_classes)

        logger.info(f"Train - Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f} | mAP: {train_map:.4f}")

        # 2. PHA XÁC THỰC (VALIDATION)
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []

        # Tắt tính toán đạo hàm để tiết kiệm bộ nhớ và tăng tốc độ
        with torch.no_grad():
            for features, labels, masks in tqdm(val_loader, desc="Xác thực", leave=False):
                features, labels, masks = features.to(device), labels.to(device), masks.to(device)
                
                outputs = model(features, masks)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * features.size(0)
                val_preds.append(outputs)
                val_targets.append(labels)

        # Tính toán độ đo tổng thể cho tập Validation
        val_loss = val_loss / len(val_loader.dataset)
        val_preds = torch.cat(val_preds, dim=0)
        val_targets = torch.cat(val_targets, dim=0)
        val_acc = calculate_accuracy(val_preds, val_targets)
        val_map = calculate_map(val_preds, val_targets, num_classes)

        logger.info(f"Val - Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f} | mAP: {val_map:.4f}")

        # Cập nhật scheduler dựa trên độ chính xác của tập validation
        scheduler.step(val_acc)
        
        # In ra Learning Rate hiện tại để tiện theo dõi
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Learning Rate hiện tại: {current_lr}")

        # 3. LƯU MÔ HÌNH TỐT NHẤT
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            logger.info(f"-> Đã lưu mô hình tốt nhất với Accuracy: {best_val_acc:.4f} tại {save_path}")

    logger.info("Hoàn tất quá trình huấn luyện!")

# ====== KHỐI LỆNH CHẠY CHÍNH ======
if __name__ == "__main__":
    # Cấu hình siêu tham số (Hyperparameters)
    FEATURE_DIR = "./data/features"
    BATCH_SIZE = 16
    # Tăng số epoch lên 30 để có đủ thời gian cho Scheduler hoạt động
    NUM_EPOCHS = 30 
    NUM_CLASSES = 14 # Thay đổi tùy vào dataset (PEC hoặc CUFED)
    LEARNING_RATE = 1e-4
    LOG_PATH = "../Docs/training_log.txt"
    SAVE_PATH = "../Release/best_peta_model.pth"

    # Thiết lập logger
    logger = setup_logger(LOG_PATH)
    logger.info("Khởi động kịch bản huấn luyện mô hình PETA")

    # Xác định thiết bị tính toán
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Thiết bị sử dụng: {device}")

    DATASET_TXT = "./data/dataset.txt"
    TRAIN_TXT = "./data/train.txt"
    
    # 2. Tạo từ điển ánh xạ (Ví dụ: {'birthday': 0, 'children_birthday': 1...})
    class_to_idx = get_class_mapping(DATASET_TXT)
    
    # 3. Load nhãn cho tập Train
    train_labels_dict = load_pec_split(TRAIN_TXT, class_to_idx)
    
    # 4. Khởi tạo Dataset chỉ với tập train
    full_train_dataset = AlbumFeatureDataset(FEATURE_DIR, train_labels_dict)
    
    # 5. Tự động trích 20% từ tập Train làm Validation
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])

    logger.info(f"Đã chia tập Train gốc | Train: {train_size} | Val: {val_size}")

    # 4. Khởi tạo DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_album_features)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_album_features)

    # Tăng num_layers lên 2 và dropout lên 0.4 để chống Overfitting
    model = PETAModel(embed_dim=2048, num_classes=NUM_CLASSES, num_heads=8, num_layers=2, dropout=0.4)
    model = model.to(device)

    # Cấu hình Loss, Optimizer và Scheduler
    criterion = nn.CrossEntropyLoss()
    
    # Dùng AdamW thay cho Adam và thêm weight_decay
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    
    # Khởi tạo Scheduler giảm Learning Rate nếu val_acc không tăng sau 3 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    # Đảm bảo thư mục Release tồn tại để lưu mô hình
    os.makedirs("../Release", exist_ok=True)

    # Bắt đầu vòng lặp huấn luyện
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler, # Truyền thêm scheduler vào hàm
        device=device,
        num_epochs=NUM_EPOCHS,
        num_classes=NUM_CLASSES,
        logger=logger,
        save_path=SAVE_PATH
    )