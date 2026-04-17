import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F 
from PIL import Image
from tqdm import tqdm

# Thư viện cho CLIP
from transformers import CLIPProcessor, CLIPModel
# Thư viện cho ResNet
import torchvision.models as models
import torchvision.transforms as transforms

def get_clip_extractor(device):
    """Khởi tạo mô hình CLIP (ViT-B/32) -> Output: 512 chiều"""
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    model.eval()
    return model, processor

def get_resnet_extractor(device):
    """Khởi tạo mô hình ResNet-50 (bỏ lớp Classification) -> Output: 2048 chiều"""
    # Tải ResNet-50 pre-trained trên ImageNet
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Cắt bỏ lớp Fully Connected cuối cùng để lấy đặc trưng (Feature Extractor)
    modules = list(resnet.children())[:-1]
    model = nn.Sequential(*modules).to(device)
    model.eval()
    
    # Tiền xử lý ảnh cho ResNet
    processor = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return model, processor

def extract_and_save_features(raw_data_dir, output_feature_dir, mode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Đang khởi tạo mô hình {mode.upper()} trên thiết bị: {device}")
    
    os.makedirs(output_feature_dir, exist_ok=True)
    
    # 1. Chọn mô hình dựa vào tham số truyền vào
    if mode == 'clip':
        model, processor = get_clip_extractor(device)
    elif mode == 'resnet':
        model, processor = get_resnet_extractor(device)
    
    album_folders = [f for f in os.listdir(raw_data_dir) if os.path.isdir(os.path.join(raw_data_dir, f))]
    
    for album_name in tqdm(album_folders, desc=f"Đang trích xuất ({mode.upper()})"):
        album_path = os.path.join(raw_data_dir, album_name)
        image_files = [img for img in os.listdir(album_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            continue
        
        album_features = []
        
        with torch.no_grad():
            for img_name in image_files:
                img_path = os.path.join(album_path, img_name)
                try:
                    img = Image.open(img_path).convert('RGB')
                    
                    # 2. Xử lý ảnh tùy theo mô hình
                    if mode == 'clip':
                        inputs = processor(images=img, return_tensors="pt").to(device)
                        outputs = model.get_image_features(**inputs)
                        
                        if not isinstance(outputs, torch.Tensor):
                            if hasattr(outputs, 'image_embeds'):
                                outputs = outputs.image_embeds
                            elif hasattr(outputs, 'pooler_output'):
                                outputs = outputs.pooler_output
                                
                    elif mode == 'resnet':
                        inputs = processor(img).unsqueeze(0).to(device)
                        # ResNet trả về tensor [1, 2048, 1, 1], cần flatten lại thành [1, 2048]
                        outputs = model(inputs).squeeze(-1).squeeze(-1)
                    
                    # 3. Chuẩn hóa Vector (L2 Norm) cho cả 2 mô hình để giữ ổn định hàm Loss
                    feature_vector = F.normalize(outputs, p=2, dim=-1)
                    feature_vector = feature_vector.squeeze(0).cpu()
                    
                    album_features.append(feature_vector)
                except Exception as e:
                    print(f"Lỗi khi xử lý ảnh {img_path}: {e}")
        
        if album_features:
            album_tensor = torch.stack(album_features)
            save_path = os.path.join(output_feature_dir, f"{album_name}.pt")
            torch.save(album_tensor, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract feature cho PETA")
    parser.add_argument('--mode', type=str, required=True, choices=['resnet', 'clip'],
                        help='Chọn mô hình (resnet, clip )')
    args = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RAW_DATA_DIRECTORY = os.path.join(BASE_DIR, "data", "raw_albums")
    
    OUTPUT_FEATURE_DIRECTORY = os.path.join(BASE_DIR, "data", "features", args.mode)
    
    print("="*50)
    print(f" Bắt đầu extract feature: {args.mode.upper()}")
    print(f" Thư mục đích: {OUTPUT_FEATURE_DIRECTORY}")
    print("="*50)
    
    extract_and_save_features(RAW_DATA_DIRECTORY, OUTPUT_FEATURE_DIRECTORY, args.mode)