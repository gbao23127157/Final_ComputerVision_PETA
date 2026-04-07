import torch
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score
import torch.nn.functional as F

def calculate_accuracy(predictions, targets):
    """
    Tính toán độ chính xác (Accuracy) tổng thể của mô hình.
    Accuracy = (Số mẫu dự đoán đúng) / (Tổng số mẫu)
    
    Tham số:
        predictions (torch.Tensor): Output của mô hình (Logits), kích thước (Batch_Size, Num_Classes).
        targets (torch.Tensor): Nhãn thực tế (Ground truth), kích thước (Batch_Size).
        
    Kết quả trả về:
        float: Giá trị Accuracy (từ 0.0 đến 1.0).
    """
    # Lấy chỉ số (index) của lớp có xác suất cao nhất làm nhãn dự đoán
    _, predicted_labels = torch.max(predictions, dim=1)
    
    # Chuyển tensor về numpy array để dùng với sklearn
    pred_np = predicted_labels.cpu().numpy()
    target_np = targets.cpu().numpy()
    
    # Sử dụng hàm accuracy_score của sklearn
    acc = accuracy_score(target_np, pred_np)
    
    return acc

def calculate_map(predictions, targets, num_classes):
    """
    Tính toán Mean Average Precision (mAP).
    Độ đo này rất quan trọng để đánh giá mô hình phân lớp khi dữ liệu mất cân bằng.
    
    Tham số:
        predictions (torch.Tensor): Output của mô hình (Logits), kích thước (Batch_Size, Num_Classes).
        targets (torch.Tensor): Nhãn thực tế, kích thước (Batch_Size).
        num_classes (int): Tổng số lớp sự kiện (C).
        
    Kết quả trả về:
        float: Giá trị mAP (từ 0.0 đến 1.0).
    """
    # Áp dụng softmax để chuyển logits thành xác suất [0, 1]
    probs = F.softmax(predictions, dim=1).detach().cpu().numpy()
    
    # Chuyển đổi nhãn thực tế sang dạng One-Hot Encoding
    target_np = targets.cpu().numpy()
    targets_one_hot = np.zeros((target_np.size, num_classes))
    targets_one_hot[np.arange(target_np.size), target_np] = 1

    # Tính Average Precision (AP) cho từng lớp, sau đó lấy trung bình (macro)
    try:
        # Hàm average_precision_score sẽ tự động bỏ qua các lớp không xuất hiện trong mảng targets
        map_score = average_precision_score(targets_one_hot, probs, average="macro")
    except ValueError:
        # Xử lý trường hợp ngoại lệ nếu batch quá nhỏ và không đủ các lớp
        map_score = 0.0
        
    return map_score