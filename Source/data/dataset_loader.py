import os
import torch
from torch.utils.data import Dataset

class AlbumFeatureDataset(Dataset):
    """
    Lớp xử lý và tải dữ liệu cho bộ dữ liệu album ảnh (PEC hoặc CUFED).
    Thay vì load ảnh thô, lớp này load các tensor đặc trưng đã được trích xuất từ trước 
    để tăng tốc độ huấn luyện cho mô hình PETA.
    """
    def __init__(self, feature_dir, labels_dict):
        """
        Hàm khởi tạo (Constructor) của lớp AlbumFeatureDataset.
        """
        self.feature_dir = feature_dir
        self.labels_dict = labels_dict
        
        # Lấy tất cả các file .pt trong thư mục
        all_files = [f for f in os.listdir(feature_dir) if f.endswith('.pt')]
        
        # CẢI TIẾN: Chỉ giữ lại những file album CÓ MẶT trong labels_dict
        # Nhờ vậy, khi truyền dict của tập Train, nó chỉ load ảnh Train.
        self.album_files = [f for f in all_files if f.split('.')[0] in self.labels_dict]

    def __len__(self):
        """
        Hàm trả về tổng số lượng album có trong tập dữ liệu.
        Hàm này bắt buộc phải override khi kế thừa từ torch.utils.data.Dataset.
        
        Kết quả trả về:
            int: Tổng số lượng album (số lượng mẫu dữ liệu).
        """
        return len(self.album_files)

    def __getitem__(self, idx):
        """
        Hàm lấy ra một mẫu dữ liệu (album và nhãn tương ứng) dựa vào chỉ số idx.
        Hàm này được DataLoader gọi tự động trong quá trình huấn luyện để tạo mini-batch.
        
        Tham số:
            idx (int): Chỉ số của album cần lấy trong danh sách self.album_files.
            
        Kết quả trả về:
            tuple: Gồm 2 phần tử:
                - features (torch.Tensor): Ma trận đặc trưng của album có kích thước (N, d).
                - label (int): Nhãn sự kiện của album đó.
        """
        # Xác định tên file của album dựa vào index
        file_name = self.album_files[idx]
        file_path = os.path.join(self.feature_dir, file_name)
        
        # Load ma trận đặc trưng F của album lên bộ nhớ
        features = torch.load(file_path)
        
        # Trích xuất tên album để tìm nhãn
        album_id = file_name.split('.')[0]
        label = self.labels_dict[album_id]
        
        return features, label