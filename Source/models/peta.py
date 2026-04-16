import torch
import torch.nn as nn
from .transformer import TransformerAttentionBlock

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    with torch.no_grad():
        l = (a - mean) / std
        u = (b - mean) / std
        tensor.normal_()
        tensor.clamp_(min=l, max=u)
        tensor.mul_(std).add_(mean)
        return tensor

class PETAModel(nn.Module):
    # Đã giảm embed_dim xuống 512 (Chuẩn của CLIP ViT-B/32)
    def __init__(self, embed_dim=512, num_classes=14, num_heads=8, num_layers=2, dropout=0.4):
        super(PETAModel, self).__init__()
        self.embed_dim = embed_dim
        
        # 1. Chỉ giữ lại [CLS] Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # [ĐÃ GỠ BỎ]: self.pos_embed = ... (Positional Encoding)
        
        self.input_dropout = nn.Dropout(p=0.2)
        
        with torch.no_grad():
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
    
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # Nối [CLS] vào chuỗi ảnh. Lúc này, x chỉ là một tập hợp các set ngữ nghĩa.
        x = torch.cat((cls_tokens, features), dim=1) 
                
        x = self.input_dropout(x)
        
        cls_mask = torch.ones(B, 1, dtype=mask.dtype, device=mask.device)
        extended_mask = torch.cat((cls_mask, mask), dim=1)
        
        for layer in self.transformer_layers:
            x, _ = layer(x, extended_mask)
            
        v_cls = x[:, 0, :]
        logits = self.mlp_head(v_cls)
         
        return logits