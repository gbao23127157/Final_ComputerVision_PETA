import torch
import torch.nn as nn
from .transformer import TransformerAttentionBlock

class PETAModel(nn.Module):
    """
    Mô hình tổng thể PETA (Photo Albums Event Recognition using Transformers Attention).
    Phiên bản Chuẩn xác 100% theo bài báo: 
    - Positional Encoding chỉ cộng vào ảnh.
    - [CLS] Token nối vào sau khi đã cộng PE.
    - Tích hợp Input Dropout chống Overfitting.
    """
    def __init__(self, embed_dim=2048, num_classes=14, num_heads=8, num_layers=2, dropout=0.4, max_len=50):
        super(PETAModel, self).__init__()
        self.embed_dim = embed_dim

        # 1. Classification Token ([CLS] Token)
        # Vector đại diện cho toàn bộ album, sẽ thu thập thông tin qua Attention
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # 2. Learnable Positional Encoding
        # ĐÃ SỬA: Kích thước bằng đúng max_len (50 ảnh), không cần cộng 1 cho [CLS]
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim))

        # 3. Input Dropout - Ép nhiễu vào dữ liệu để chống học vẹt (Overfitting)
        self.input_dropout = nn.Dropout(p=0.2)

        # 4. Khối Lọc nhiễu bằng Transformer Attention
        self.transformer_layers = nn.ModuleList([
            TransformerAttentionBlock(embed_dim, num_heads, dropout) 
            for _ in range(num_layers)
        ])
        
        # 5. Mạng tổng hợp và Phân loại (MLP Head)
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, features, mask):
        """
        Luồng truyền dữ liệu tiến (Forward Pass).
        """
        B, N, _ = features.shape
        
        # Bước 1: Cộng Positional Encoding cho 50 ảnh TRƯỚC
        # (Theo đúng logic bài báo: z_s = x_s + e_pos_s)
        x_images = features + self.pos_embed[:, :N, :]
        
        # Bước 2: Chuẩn bị [CLS] Token (Nhân bản để khớp kích thước Batch)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        # Bước 3: Nối [CLS] vào ĐẦU chuỗi ảnh (SAU KHI ĐÃ CỘNG VỊ TRÍ)
        # Kích thước x lúc này là (Batch, N + 1, embed_dim)
        x = torch.cat((cls_tokens, x_images), dim=1)
        
        # Bước 4: Áp dụng Dropout cho đầu vào trước khi đưa vào khối Transformer
        x = self.input_dropout(x)
        
        # Bước 5: Cập nhật Mask (Thêm 1 vị trí hợp lệ cho [CLS] ở cột đầu tiên)
        cls_mask = torch.ones(B, 1, dtype=mask.dtype, device=mask.device)
        extended_mask = torch.cat((cls_mask, mask), dim=1)
        
        # Bước 6: Đi qua các lớp Transformer Attention
        for layer in self.transformer_layers:
            x, attn_weights = layer(x, extended_mask)
        
        # Bước 7: Trích xuất vector đại diện album (Chỉ lấy [CLS] token ở Index 0)
        v_album = x[:, 0, :]
        
        # Bước 8: Phân loại sự kiện
        logits = self.mlp_head(v_album)
         
        return logits