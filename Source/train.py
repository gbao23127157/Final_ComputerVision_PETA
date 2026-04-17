import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset_loader import AlbumFeatureDataset
from utils.metrics import calculate_accuracy, calculate_map
from utils.logger import setup_logger

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
        
        # Trộn ngẫu nhiên toàn bộ ảnh trước khi lấy sample
        indices = torch.randperm(n)
        features = features[indices]
        
        if n >= num_samples:
            sampled_features = features[:num_samples]
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

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, num_classes, logger, save_path):
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        logger.info(f"--- Epoch {epoch + 1}/{num_epochs} ---")
        
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []

        for features, labels, masks in tqdm(train_loader, desc="Huấn luyện", leave=False):
            features, labels, masks = features.to(device), labels.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(features, masks)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            train_preds.append(outputs.detach())
            train_targets.append(labels.detach())

        train_loss = train_loss / len(train_loader.dataset)
        train_preds = torch.cat(train_preds, dim=0)
        train_targets = torch.cat(train_targets, dim=0)
        train_acc = calculate_accuracy(train_preds, train_targets)
        train_map = calculate_map(train_preds, train_targets, num_classes)

        logger.info(f"Train - Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f} | mAP: {train_map:.4f}")

        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for features, labels, masks in tqdm(val_loader, desc="Xác thực", leave=False):
                features, labels, masks = features.to(device), labels.to(device), masks.to(device)
                
                outputs = model(features, masks)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * features.size(0)
                val_preds.append(outputs)
                val_targets.append(labels)

        val_loss = val_loss / len(val_loader.dataset)
        val_preds = torch.cat(val_preds, dim=0)
        val_targets = torch.cat(val_targets, dim=0)
        val_acc = calculate_accuracy(val_preds, val_targets)
        val_map = calculate_map(val_preds, val_targets, num_classes)

        logger.info(f"Valid - Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f} | mAP: {val_map:.4f}")

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Learning Rate hiện tại: {current_lr:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            logger.info(f"-> Đã lưu mô hình tốt nhất với Accuracy: {best_val_acc:.4f} tại {save_path}")

    logger.info("Hoàn tất quá trình huấn luyện!")

if __name__ == "__main__":
    # Thiết lập tham số dòng lệnh
    parser = argparse.ArgumentParser(description="Huấn luyện các phiên bản mô hình PETA")
    parser.add_argument('--mode', type=str, default='cross', 
                        choices=['baseline', 'peta_base', 'peta_clip', 'peta_cross'],
                        help='Chọn phiên bản mô hình để huấn luyện')
    args = parser.parse_args()

    BATCH_SIZE = 16 
    NUM_EPOCHS = 30 
    NUM_CLASSES = 14
    LEARNING_RATE = 0.01
    NUM_SAMPLES = 50 

    os.makedirs("./weights", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    LOG_PATH = f"./logs/training_log_{args.mode}.txt"

    logger = setup_logger(LOG_PATH)
    logger.info(f"Khởi động huấn luyện CHẾ ĐỘ: {args.mode.upper()}")

    # ĐIỀU PHỐI MÔ HÌNH VÀ DỮ LIỆU
    if args.mode == 'baseline':
        FEATURE_DIR = "./data/features/resnet"
        SAVE_PATH = "./weights/baseline_weights.pth"
        model = BaselineModel(embed_dim=2048, num_classes=NUM_CLASSES)
        
    elif args.mode == 'peta_base':
        FEATURE_DIR = "./data/features/resnet"
        SAVE_PATH = "./weights/peta_base_weights.pth"
        model = PETA(embed_dim=2048, num_classes=NUM_CLASSES, num_heads=8, num_layers=2, max_len=NUM_SAMPLES)
        
    elif args.mode == 'peta_clip':
        FEATURE_DIR = "./data/features/clip"
        SAVE_PATH = "./weights/peta_clip_weights.pth"
        model = PETAClip(embed_dim=512, num_classes=NUM_CLASSES, num_heads=8, num_layers=2)
        
    elif args.mode == 'peta_cross':
        FEATURE_DIR = "./data/features/clip"
        SAVE_PATH = "./weights/peta_cross_weights.pth"
        model = PETACross(embed_dim=512, num_classes=NUM_CLASSES, num_heads=8, num_layers=2)

    logger.info(f"Sử dụng thư mục đặc trưng: {FEATURE_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    DATASET_TXT = "./data/dataset.txt"
    TRAIN_TXT = "./data/train.txt" 
    TEST_TXT = "./data/test.txt"   
    
    class_to_idx = get_class_mapping(DATASET_TXT)
    train_labels_dict = load_pec_split(TRAIN_TXT, class_to_idx)
    test_labels_dict = load_pec_split(TEST_TXT, class_to_idx)
    
    train_dataset = AlbumFeatureDataset(FEATURE_DIR, train_labels_dict)
    val_dataset = AlbumFeatureDataset(FEATURE_DIR, test_labels_dict)

    logger.info(f"Dữ liệu ép chuẩn: Train: {len(train_dataset)} albums | Test: {len(val_dataset)} albums")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=fixed_sample_collate)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=fixed_sample_collate)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25], gamma=0.1)

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=NUM_EPOCHS,
        num_classes=NUM_CLASSES,
        logger=logger,
        save_path=SAVE_PATH
    )