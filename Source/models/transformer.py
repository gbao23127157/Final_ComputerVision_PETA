import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerAttentionBlock(nn.Module):
    """
    Khối Transformer Attention cốt lõi.
    [CẬP NHẬT MAX SETTINGS]: Tắt dropout ở MultiheadAttention để trọng số attention sắc nét hơn.
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(TransformerAttentionBlock, self).__init__()
        
        # [ĐÃ SỬA]: Set dropout=0.0 cho MultiheadAttention
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0)
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Vẫn giữ dropout ở FFN để tránh overfit
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x_permuted = x.permute(1, 0, 2)
        key_padding_mask = (mask == 0) if mask is not None else None
        
        attn_output, attn_weights = self.multihead_attn(
            query=x_permuted, 
            key=x_permuted, 
            value=x_permuted,
            key_padding_mask=key_padding_mask
        )
        
        x_permuted = self.norm(x_permuted + self.dropout(attn_output))
        ffn_output = self.ffn(x_permuted)
        x_permuted = self.norm(x_permuted + self.dropout(ffn_output))
        
        return x_permuted.permute(1, 0, 2), attn_weights