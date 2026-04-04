import torch
import torch.nn.functional as F

def pad_album_features(batch):
    """
    Hàm xử lý gom cụm (collate_fn) để giải quyết vấn đề các album có số lượng ảnh khác nhau.
    Các album ngắn hơn sẽ được thêm (padding) các vector 0 để bằng chiều dài album dài nhất trong batch.
    
    Tham số:
        batch (list): Danh sách các mẫu lấy từ AlbumFeatureDataset, 
                      mỗi mẫu là một tuple (features, label).
                      
    Kết quả trả về:
        padded_features (torch.Tensor): Tensor kích thước (Batch_Size, Max_N, d)
        labels (torch.Tensor): Tensor chứa các nhãn tương ứng.
        mask (torch.Tensor): Tensor nhị phân đánh dấu vị trí ảnh thực tế (1) và padding (0).
    """
    # Tách danh sách đặc trưng và nhãn từ batch
    features = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    
    # Tìm số lượng ảnh lớn nhất (N_max) trong batch hiện tại
    max_n = max([f.size(0) for f in features])
    feature_dim = features[0].size(1) # d (ví dụ 2048)
    
    padded_features = []
    masks = []
    
    for f in features:
        n = f.size(0)
        # Tính toán lượng padding cần thiết: (0, 0) cho chiều d, (0, max_n - n) cho chiều N
        pad_size = (0, 0, 0, max_n - n)
        
        # Thực hiện padding với giá trị 0
        padded_f = F.pad(f, pad_size, mode='constant', value=0)
        padded_features.append(padded_f)
        
        # Tạo mask để mô hình Transformer biết đâu là ảnh thật, đâu là ảnh padding
        # (Quan trọng cho cơ chế Self-Attention trong kiến trúc PETA)
        m = torch.zeros(max_n)
        m[:n] = 1
        masks.append(m)
        
    # Gộp danh sách thành các Tensor duy nhất
    padded_features = torch.stack(padded_features)
    masks = torch.stack(masks)
    
    return padded_features, labels, masks

def fixed_sample_preprocess(features, num_samples=50):
    """
    Phương pháp tiền xử lý thay thế: Cố định số lượng ảnh cho mỗi album.
    Nếu album > num_samples: Lấy mẫu ngẫu nhiên.
    Nếu album < num_samples: Lặp lại ảnh hoặc padding.
    
    Tham số:
        features (torch.Tensor): Đặc trưng album gốc (N, d).
        num_samples (int): Số lượng ảnh cố định muốn lấy.
        
    Kết quả trả về:
        torch.Tensor: Đặc trưng sau khi lấy mẫu (num_samples, d).
    """
    n = features.size(0)
    
    if n >= num_samples:
        # Lấy mẫu ngẫu nhiên không lặp lại
        indices = torch.randperm(n)[:num_samples]
        return features[indices]
    else:
        # Nếu thiếu ảnh, tiến hành padding bằng vector 0
        pad_size = (0, 0, 0, num_samples - n)
        return F.pad(features, pad_size, mode='constant', value=0)