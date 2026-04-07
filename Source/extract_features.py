import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm # Thư viện tạo thanh tiến trình (progress bar) trực quan

def get_feature_extractor(device):
    """
    Khởi tạo mô hình CNN (ResNet50) đã được huấn luyện sẵn (pre-trained) để làm bộ trích xuất đặc trưng.
    Lớp phân loại (fully connected layer) cuối cùng được gỡ bỏ vì chỉ cần lấy vector đặc trưng.
    
    Tham số:
        device (torch.device): Thiết bị sẽ chạy mô hình (CPU hoặc CUDA/GPU).
        
    Kết quả trả về:
        torch.nn.Module: Mô hình ResNet50 đã được tinh chỉnh để xuất ra vector 2048 chiều.
    """
    # Tải mô hình ResNet50 với bộ trọng số tốt nhất được huấn luyện trên ImageNet
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # Thay thế lớp Fully Connected (fc) cuối cùng bằng hàm Identity 
    # để output trả về là vector đặc trưng (kích thước 2048) thay vì xác suất 1000 lớp
    model.fc = nn.Identity()
    
    # Đẩy mô hình lên GPU (nếu có) và thiết lập chế độ evaluate (eval)
    # Chế độ eval() giúp tắt Dropout và cố định BatchNorm, đảm bảo tính nhất quán khi trích xuất
    model = model.to(device)
    model.eval()
    
    return model

def get_image_transforms():
    """
    Khởi tạo chuỗi các phép biến đổi (transform) để tiền xử lý ảnh trước khi đưa vào mô hình CNN.
    Các thông số chuẩn hóa này dựa trên chuẩn của tập dữ liệu ImageNet.
    
    Kết quả trả về:
        torchvision.transforms.Compose: Chuỗi các phép tiền xử lý ảnh.
    """
    preprocess = transforms.Compose([
        transforms.Resize(256),              # Thay đổi kích thước cạnh nhỏ nhất về 256 pixel
        transforms.CenterCrop(224),          # Cắt lấy vùng trung tâm kích thước 224x224
        transforms.ToTensor(),               # Chuyển đổi ảnh (PIL) thành Tensor và scale giá trị về [0, 1]
        transforms.Normalize(                # Chuẩn hóa tensor với mean và std của ImageNet
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    return preprocess

def extract_and_save_features(raw_data_dir, output_feature_dir):
    """
    Duyệt qua toàn bộ thư mục chứa album ảnh, dùng mô hình CNN để trích xuất đặc trưng
    và lưu kết quả của mỗi album thành một file tensor (.pt).
    
    Tham số:
        raw_data_dir (str): Đường dẫn đến thư mục chứa ảnh gốc (Mỗi thư mục con là 1 album).
        output_feature_dir (str): Đường dẫn đến thư mục sẽ lưu các file tensor đặc trưng.
    """
    # Khởi tạo thiết bị và tạo thư mục lưu trữ nếu chưa có
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị tính toán: {device}")
    
    os.makedirs(output_feature_dir, exist_ok=True)
    
    model = get_feature_extractor(device)
    preprocess = get_image_transforms()
    
    # Lấy danh sách các thư mục album con trong thư mục chứa dữ liệu gốc
    album_folders = [f for f in os.listdir(raw_data_dir) if os.path.isdir(os.path.join(raw_data_dir, f))]
    
    # Vòng lặp duyệt qua từng album với thanh tiến trình
    for album_name in tqdm(album_folders, desc="Đang trích xuất đặc trưng các album"):
        album_path = os.path.join(raw_data_dir, album_name)
        image_files = [img for img in os.listdir(album_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            continue
        
        album_features = []
        
        # Tắt cơ chế tính đạo hàm (gradient) để tiết kiệm bộ nhớ và tăng tốc độ
        with torch.no_grad():
            for img_name in image_files:
                img_path = os.path.join(album_path, img_name)
                try:
                    # Mở ảnh và chuyển về hệ màu RGB 
                    img = Image.open(img_path).convert('RGB')
                    
                    # Áp dụng tiền xử lý và thêm chiều batch (từ C,H,W thành 1,C,H,W)
                    input_tensor = preprocess(img).unsqueeze(0).to(device)
                    
                    # Trích xuất đặc trưng và đưa tensor về lại CPU
                    feature_vector = model(input_tensor).squeeze(0).cpu()
                    album_features.append(feature_vector)
                except Exception as e:
                    print(f"Lỗi khi xử lý ảnh {img_path}: {e}")
        
        # Gộp danh sách các vector 1D thành một ma trận 2D kích thước (N, d)
        if album_features:
            album_tensor = torch.stack(album_features)
            
            # Lưu ma trận đặc trưng thành file .pt tương ứng với tên album
            save_path = os.path.join(output_feature_dir, f"{album_name}.pt")
            torch.save(album_tensor, save_path)

if __name__ == "__main__":
    # Lấy thư mục chứa file extract_features.py hiện tại 
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    RAW_DATA_DIRECTORY = os.path.join(BASE_DIR, "data", "raw_albums")
    OUTPUT_FEATURE_DIRECTORY = os.path.join(BASE_DIR, "data", "features")
    
    print("Bắt đầu quá trình trích xuất đặc trưng...")
    extract_and_save_features(RAW_DATA_DIRECTORY, OUTPUT_FEATURE_DIRECTORY)
    print("Hoàn tất! Các file tensor đã được lưu tại:", OUTPUT_FEATURE_DIRECTORY)