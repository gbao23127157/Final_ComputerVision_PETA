import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerAttentionBlock(nn.Module):
    """
    Khối Transformer Attention cốt lõi của kiến trúc PETA.
    Nhiệm vụ: Tính toán sự tương quan giữa các ảnh trong album để lọc nhiễu và khuếch đại các đặc trưng quan trọng.
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        """
        Khởi tạo khối Attention.
        
        Tham số:
            embed_dim (int): Số chiều của vector đặc trưng (ví dụ: 2048 với ResNet50).
            num_heads (int): Số lượng "đầu" chú ý để học các mối quan hệ khác nhau.
            dropout (float): Tỉ lệ loại bỏ nơ-ron để chống Overfitting.
        """
        super(TransformerAttentionBlock, self).__init__()
        
        # Sử dụng MultiheadAttention có sẵn của PyTorch để tối ưu hiệu năng
        # embed_dim tương ứng với d, num_heads tương ứng với số luồng xử lý song song
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        
        # Lớp chuẩn hóa (Layer Normalization) giúp ổn định quá trình huấn luyện
        self.norm = nn.LayerNorm(embed_dim)
        
        # Mạng Feed Forward nhỏ (MLP) sau khi chú ý
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Luồng xử lý dữ liệu (Forward pass).
        
        Tham số:
            x (torch.Tensor): Tensor đặc trưng album kích thước (Batch_Size, N, d).
            mask (torch.Tensor): Mask nhị phân để bỏ qua các ảnh padding (từ preprocess.py).
            
        Kết quả trả về:
            output (torch.Tensor): Đặc trưng đã được lọc nhiễu và cập nhật ngữ cảnh (Batch_Size, N, d).
        """
        # Bước 1: Chuẩn bị định dạng dữ liệu và Mask
        # Chuyển đổi tensor sang (Sequence_Length, Batch, Dimension) để phù hợp với PyTorch MHA
        x_permuted = x.permute(1, 0, 2)
        
        # Tạo key_padding_mask từ mask (Ảnh thật = False, Padding = True)
        key_padding_mask = (mask == 0) if mask is not None else None
        
        # Bước 2: Tính toán Self-Attention (Q, K, V đều là x_permuted)
        # attn_output chính là F' 
        attn_output, attn_weights = self.multihead_attn(
            query=x_permuted, 
            key=x_permuted, 
            value=x_permuted,
            key_padding_mask=key_padding_mask
        )
        
        # Residual connection và Layer Normalization
        x_permuted = self.norm(x_permuted + self.dropout(attn_output))
        
        # Bước 3: Đi qua mạng Feed Forward
        ffn_output = self.ffn(x_permuted)
        x_permuted = self.norm(x_permuted + self.dropout(ffn_output))
        
        # Trả về định dạng ban đầu (Batch_Size, N, d)
        return x_permuted.permute(1, 0, 2), attn_weights