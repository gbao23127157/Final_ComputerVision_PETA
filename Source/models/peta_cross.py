import torch
import torch.nn as nn

class CrossAttentionPooling(nn.Module):
    """
    Khối Cross-Attention: [CLS] Token đóng vai trò là Query (Q).
    N bức ảnh đóng vai trò là Key (K) và Value (V).
    Độ phức tạp O(N) thay vì O(N^2).
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(CrossAttentionPooling, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm_ffn = nn.LayerNorm(embed_dim)

    def forward(self, query, keys, key_padding_mask=None):
        attn_output, attn_weights = self.multihead_attn(
            query=query, 
            key=keys, 
            value=keys,
            key_padding_mask=key_padding_mask
        )
        
        q_norm = self.norm(query + self.dropout(attn_output))
        ffn_out = self.ffn(q_norm)
        out = self.norm_ffn(q_norm + self.dropout(ffn_out))
        
        return out, attn_weights

class PETAModel(nn.Module):
    def __init__(self, embed_dim=512, num_classes=14, num_heads=8, num_layers=2, dropout=0.4):
        super(PETAModel, self).__init__()
        self.embed_dim = embed_dim
        
        # Khởi tạo Global Query (CLS Token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.input_dropout = nn.Dropout(p=0.2)
        
        # Khối Cross-Attention thay cho Self-Attention
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionPooling(embed_dim, num_heads, dropout) 
            for _ in range(num_layers)
        ])
        
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, features, mask):
        B, N, _ = features.shape
        features = self.input_dropout(features)
        
        # [CLS] Token lấy từ tham số mô hình, expand ra theo Batch Size
        query = self.cls_token.expand(B, -1, -1)
        
        # Tạo mask: 0 là hợp lệ, 1 là padding (theo chuẩn của nn.MultiheadAttention mới)
        key_padding_mask = (mask == 0) if mask is not None else None
        
        # Token CLS (query) đi "soi" toàn bộ N bức ảnh (features)
        for layer in self.cross_attn_layers:
            query, _ = layer(query, features, key_padding_mask)
            
        # Trích xuất vector [CLS] cuối cùng
        v_cls = query.squeeze(1) # Từ [B, 1, 512] -> [B, 512]
        logits = self.mlp_head(v_cls)
         
        return logits