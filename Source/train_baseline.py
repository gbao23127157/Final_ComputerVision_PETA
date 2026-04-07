import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset_loader import AlbumFeatureDataset
from models.baseline import BaselineModel
from utils.metrics import calculate_accuracy, calculate_map
from utils.logger import setup_logger

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

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, num_classes, logger, save_path):
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
        logger.info(f"Val - Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f} | mAP: {val_map:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            logger.info(f"-> Đã lưu mô hình tốt nhất với Accuracy: {best_val_acc:.4f} tại {save_path}")

if __name__ == "__main__":
    FEATURE_DIR = "./data/features"
    BATCH_SIZE = 16
    NUM_EPOCHS = 20 
    NUM_CLASSES = 14 
    LEARNING_RATE = 1e-4
    LOG_PATH = "../Docs/training_log_baseline.txt" 
    SAVE_PATH = "../Release/best_baseline_model.pth"

    logger = setup_logger(LOG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATASET_TXT = "./data/dataset.txt"
    TRAIN_TXT = "./data/train.txt"
    class_to_idx = get_class_mapping(DATASET_TXT)
    train_labels_dict = load_pec_split(TRAIN_TXT, class_to_idx)
    full_train_dataset = AlbumFeatureDataset(FEATURE_DIR, train_labels_dict)
    
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=fixed_sample_collate)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=fixed_sample_collate)

    model = BaselineModel(embed_dim=2048, num_classes=NUM_CLASSES, dropout=0.3)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    os.makedirs("../Release", exist_ok=True)

    train_model(model, train_loader, val_loader, criterion, optimizer, device, NUM_EPOCHS, NUM_CLASSES, logger, SAVE_PATH)