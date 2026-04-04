import torch
import torch.nn as nn
from .transformer import TransformerAttentionBlock

class PETAModel(nn.Module):
    """
    Mô hình tổng thể PETA (Photo Albums Event Recognition using Transformers Attention).
    Kết hợp cơ chế Attention để lọc nhiễu và MLP để phân loại sự kiện của album ảnh.
    """
    def __init__(self, embed_dim=2048, num_classes=14, num_heads=8, num_layers=1, dropout=0.1):
        """
        Khởi tạo kiến trúc mô hình PETA.
        
        Tham số:
            embed_dim (int): Số chiều của vector đặc trưng từ CNN (Mặc định 2048 của ResNet50).
            num_classes (int): Số lượng lớp sự kiện cần phân loại (Tập CUFED thường có 14, PEC có 14).
            num_heads (int): Số luồng chú ý (heads) trong khối Transformer.
            num_layers (int): Số lượng lớp Transformer xếp chồng lên nhau.
            dropout (float): Tỉ lệ ngắt kết nối để giảm Overfitting.
        """
        super(PETAModel, self).__init__()
        self.embed_dim = embed_dim
        
        # Bước 2: Khối Lọc nhiễu bằng Transformer Attention
        # Có thể xếp chồng nhiều lớp Transformer để học ngữ cảnh sâu hơn
        self.transformer_layers = nn.ModuleList([
            TransformerAttentionBlock(embed_dim, num_heads, dropout) 
            for _ in range(num_layers)
        ])
        
        # Bước 3: Mạng tổng hợp và Phân loại (MLP Head)
        # Giảm chiều dữ liệu dần dần từ embed_dim xuống num_classes
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def masked_average_pooling(self, features, mask):
        """
        Thực hiện tính trung bình cộng (Average Pooling) nhưng bỏ qua các vector padding.
        Đảm bảo vector đại diện cho album (V_album) không bị nhiễu bởi các giá trị 0.
        
        Tham số:
            features (torch.Tensor): Đặc trưng đầu ra từ Transformer, kích thước (Batch, N, d).
            mask (torch.Tensor): Mask nhị phân từ dataloader, kích thước (Batch, N).
            
        Kết quả trả về:
            v_album (torch.Tensor): Vector đại diện cho toàn bộ album, kích thước (Batch, d).
        """
        # Mở rộng chiều của mask từ (Batch, N) thành (Batch, N, 1) để nhân với features
        mask_expanded = mask.unsqueeze(-1)
        
        # Chỉ giữ lại các đặc trưng của ảnh thật, ảnh padding sẽ biến thành 0
        masked_features = features * mask_expanded
        
        # Tính tổng các vector đặc trưng của ảnh thật trong mỗi album
        sum_features = torch.sum(masked_features, dim=1)
        
        # Đếm số lượng ảnh thật trong mỗi album (chống chia cho 0 bằng clamp)
        valid_counts = torch.clamp(torch.sum(mask_expanded, dim=1), min=1.0)
        
        # Chia tổng cho số lượng để ra trung bình
        v_album = sum_features / valid_counts
        
        return v_album

    def forward(self, features, mask):
        """
        Luồng truyền dữ liệu tiến (Forward Pass) của toàn bộ kiến trúc PETA.
        
        Tham số:
            features (torch.Tensor): Ma trận đặc trưng album gốc, kích thước (Batch, N, d).
            mask (torch.Tensor): Mask nhị phân đánh dấu vị trí ảnh thật.
            
        Kết quả trả về:
            logits (torch.Tensor): Phân phối dự đoán (chưa qua softmax), kích thước (Batch, C).
        """
        x = features
        
        # Đi qua tuần tự các lớp Transformer Attention để lọc nhiễu
        for layer in self.transformer_layers:
            x, attn_weights = layer(x, mask)
        
        # x lúc này là F' đã được làm sạch và mang ngữ cảnh toàn cục
        
        # Bước 3: Tổng hợp thành 1 vector duy nhất cho cả album
        v_album = self.masked_average_pooling(x, mask)
        
        # Đưa qua mạng MLP để phân lớp
        logits = self.mlp_head(v_album)
        
        # Lưu ý: Ta trả về logits thô thay vì dùng hàm softmax ở đây.
        # Lý do là trong script train.py, hàm nn.CrossEntropyLoss() của PyTorch 
        # đã tự động bao gồm cả phép tính Log-Softmax bên trong nó.
        return logits