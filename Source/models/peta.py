import torch
import torch.nn as nn
import math
from .transformer import TransformerAttentionBlock

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """
    Hàm khởi tạo Truncated Normal (Cắt cụt phân phối chuẩn).
    Được sử dụng trong Vision Transformer (ViT) và code gốc của PETA
    để giúp mô hình ổn định trong những epoch đầu tiên.
    """
    with torch.no_grad():
        l = (a - mean) / std
        u = (b - mean) / std
        
        # Lấy mẫu ngẫu nhiên từ phân phối chuẩn
        tensor.normal_()
        
        # Cắt các giá trị nằm ngoài khoảng [a, b]
        tensor.clamp_(min=l, max=u)
        
        # Dịch chuyển và co giãn lại
        tensor.mul_(std).add_(mean)
        return tensor

class PETAModel(nn.Module):
    def __init__(self, embed_dim=2048, num_classes=14, num_heads=8, num_layers=2, dropout=0.4, max_len=50):
        super(PETAModel, self).__init__()
        self.embed_dim = embed_dim
        
        # 1. Khởi tạo [CLS] Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 2. Khởi tạo Positional Encoding (ĐÃ SỬA: Kích thước là max_len + 1 để khớp cả [CLS])
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len + 1, embed_dim))
        
        self.input_dropout = nn.Dropout(p=0.2)
        
        # Áp dụng khởi tạo Truncated Normal (trick từ code gốc)
        with torch.no_grad():
            trunc_normal_(self.pos_embed, std=.02)
            trunc_normal_(self.cls_token, std=.02)
            
        self.transformer_layers = nn.ModuleList([
            TransformerAttentionBlock(embed_dim, num_heads, dropout) 
            for _ in range(num_layers)
        ])
        
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
        # Khởi tạo Linear Layers với std=0.02
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, features, mask):
        B, N, _ = features.shape
        
        # [SỬA THEO TRÌNH TỰ GỐC]: Nối [CLS] VÀO TRƯỚC, SAU ĐÓ MỚI CỘNG VỊ TRÍ
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, features), dim=1) # Nối [CLS] vào chuỗi N ảnh
        
        # Cộng Positional Encoding cho toàn bộ (N+1) tokens
        # x.shape = (B, N+1, embed_dim)
        x = x + self.pos_embed[:, :(N + 1), :]
        
        x = self.input_dropout(x)
        
        cls_mask = torch.ones(B, 1, dtype=mask.dtype, device=mask.device)
        extended_mask = torch.cat((cls_mask, mask), dim=1)
        
        for layer in self.transformer_layers:
            x, _ = layer(x, extended_mask)
            
        v_cls = x[:, 0, :]
        logits = self.mlp_head(v_cls)
         
        return logits