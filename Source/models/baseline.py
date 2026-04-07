import torch
import torch.nn as nn

class BaselineModel(nn.Module):
    """
    Mô hình cơ sở (Baseline): 
    Chỉ dùng Average Pooling gộp đặc trưng tất cả các ảnh lại, KHÔNG CÓ khối Transformer.
    """
    def __init__(self, embed_dim=2048, num_classes=14, dropout=0.3):
        super(BaselineModel, self).__init__()
        
        # Khối phân loại (MLP Head) - Kiến trúc giống hệt PETA để so sánh công bằng
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, features, masks):
        # features kích thước: (Batch_size, N_ảnh, 2048)
        # masks kích thước: (Batch_size, N_ảnh)
        
        # 1. Masked Average Pooling (Gộp cào bằng)
        # Mở rộng mask để nhân với features: (Batch, N, 1)
        mask_expanded = masks.unsqueeze(-1) 
        valid_features = features * mask_expanded
        
        # Tính tổng các ảnh hợp lệ
        sum_features = valid_features.sum(dim=1) 
        
        # Đếm số lượng ảnh hợp lệ (tránh chia cho 0 bằng cách dùng clamp)
        valid_counts = masks.sum(dim=1, keepdim=True).clamp(min=1e-9) 
        
        # Chia trung bình -> Vector đại diện cho album: (Batch, 2048)
        avg_pool_features = sum_features / valid_counts 
        
        # 2. Đưa qua lớp phân loại
        logits = self.classifier(avg_pool_features)
        
        return logits