import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(TransformerAttentionBlock, self).__init__()
        
        # Giữ dropout=0.0 cho MultiheadAttention như đã phân tích
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # PyTorch MultiheadAttention yêu cầu input: (Sequence_Length, Batch_Size, Embed_Dim)
        x_permuted = x.permute(1, 0, 2)
        
        # Key Padding Mask: Chỗ nào mask == 0 thì là True (bị bỏ qua)
        key_padding_mask = (mask == 0) if mask is not None else None
        
        attn_output, attn_weights = self.multihead_attn(
            query=x_permuted, 
            key=x_permuted, 
            value=x_permuted,
            key_padding_mask=key_padding_mask
        )
        
        # Trình tự chuẩn: (x + dropout(attn)) -> Norm
        x_permuted = self.norm1(x_permuted + self.dropout1(attn_output))
        
        ffn_output = self.ffn(x_permuted)
        x_permuted = self.norm2(x_permuted + self.dropout2(ffn_output))
        
        # Trả lại kích thước (Batch_Size, Sequence_Length, Embed_Dim)
        return x_permuted.permute(1, 0, 2), attn_weights