import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Cấu trúc import từ các module đã xây dựng
from data.dataset_loader import AlbumFeatureDataset
from data.preprocess import pad_album_features
from models.peta import PETAModel
from utils.metrics import calculate_accuracy, calculate_map
from utils.logger import setup_logger

def load_labels_dict(label_file_path):
    """
    Đọc file văn bản chứa nhãn thực tế của tập dữ liệu.
    Định dạng giả định mỗi dòng: <tên_album> <khoảng_trắng> <mã_nhãn_số_nguyên>
    Ví dụ: album_001 3
    
    Tham số:
        label_file_path (str): Đường dẫn đến file txt/csv chứa nhãn.
        
    Kết quả trả về:
        dict: Ánh xạ từ tên album sang nhãn (int).
    """
    labels_dict = {}
    
    try:
        with open(label_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split() # Cắt dòng theo khoảng trắng
                if len(parts) >= 2:
                    album_id = parts[0]
                    # Chuyển nhãn sự kiện thành số nguyên
                    label = int(parts[1]) 
                    labels_dict[album_id] = label
                    
        print(f"Đã tải thành công {len(labels_dict)} nhãn từ file {label_file_path}")
    except Exception as e:
        print(f"Lỗi khi đọc file nhãn: {e}")
        
    return labels_dict

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, num_classes, logger, save_path):
    """
    Hàm thực hiện vòng lặp huấn luyện và đánh giá mô hình qua nhiều epoch.
    Lưu lại mô hình có độ chính xác trên tập validation cao nhất.
    
    Tham số:
        model (nn.Module): Kiến trúc mạng PETA.
        train_loader (DataLoader): Trình tải dữ liệu huấn luyện.
        val_loader (DataLoader): Trình tải dữ liệu xác thực (validation).
        criterion (Loss Function): Hàm tính sai số (CrossEntropyLoss).
        optimizer (Optimizer): Thuật toán tối ưu (ví dụ: Adam).
        device (torch.device): Thiết bị chạy (CPU/GPU).
        num_epochs (int): Số chu kỳ huấn luyện.
        num_classes (int): Số lượng lớp sự kiện.
        logger (logging.Logger): Công cụ ghi log.
        save_path (str): Đường dẫn lưu file trọng số tốt nhất (.pth).
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

        logger.info(f"Val   - Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f} | mAP: {val_map:.4f}")

        # 3. LƯU MÔ HÌNH TỐT NHẤT
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            logger.info(f"-> Đã lưu mô hình tốt nhất với Accuracy: {best_val_acc:.4f} tại {save_path}")

    logger.info("Hoàn tất quá trình huấn luyện!")

# ====== KHỐI LỆNH CHẠY CHÍNH (MAIN ENTRY POINT) ======
if __name__ == "__main__":
    # Cấu hình siêu tham số (Hyperparameters)
    FEATURE_DIR = "./data/features"
    BATCH_SIZE = 16
    NUM_EPOCHS = 20
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

    LABEL_FILE = "./data/labels.txt"
    labels_dict = load_labels_dict(LABEL_FILE)

    # Nếu chưa trích xuất đặc trưng, thông báo cho người dùng
    if not os.path.exists(FEATURE_DIR) or len(os.listdir(FEATURE_DIR)) == 0:
        logger.error("Không tìm thấy file đặc trưng! Vui lòng chạy file extract_features.py trước.")
        exit(1)

    # Khởi tạo Dataset
    dataset = AlbumFeatureDataset(FEATURE_DIR, labels_dict)
    
    # Chia dữ liệu theo tỷ lệ 80% Train, 20% Validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    logger.info(f"Tổng số album: {len(dataset)} | Train: {train_size} | Val: {val_size}")

    # Khởi tạo DataLoader, LƯU Ý truyền collate_fn=pad_album_features để xử lý album khác độ dài
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_album_features)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_album_features)

    # Khởi tạo Mô hình
    model = PETAModel(embed_dim=2048, num_classes=NUM_CLASSES, num_heads=8, num_layers=1, dropout=0.3)
    model = model.to(device)

    # Cấu hình Loss và Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Đảm bảo thư mục Release tồn tại để lưu mô hình
    os.makedirs("../Release", exist_ok=True)

    # Bắt đầu vòng lặp huấn luyện
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=NUM_EPOCHS,
        num_classes=NUM_CLASSES,
        logger=logger,
        save_path=SAVE_PATH
    )