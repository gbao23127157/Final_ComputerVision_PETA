import os
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
# Thêm thư viện HuggingFace
from transformers import CLIPProcessor, CLIPModel

def get_clip_extractor(device):
    """
    Sử dụng OpenAI CLIP (ViT-B/32) thay cho ResNet-50.
    CLIP giúp trích xuất trực tiếp các khái niệm ngữ nghĩa thay vì chỉ vật thể.
    """
    # Tải mô hình CLIP và bộ tiền xử lý ảnh
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    
    model.eval()
    return model, processor
import os
import torch
import torch.nn as nn
import torch.nn.functional as F  # <--- THÊM DÒNG NÀY Ở ĐẦU FILE
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# ... (Hàm get_clip_extractor giữ nguyên) ...

def extract_and_save_features(raw_data_dir, output_feature_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị tính toán: {device} với mô hình CLIP")
    
    os.makedirs(output_feature_dir, exist_ok=True)
    
    model, processor = get_clip_extractor(device)
    
    album_folders = [f for f in os.listdir(raw_data_dir) if os.path.isdir(os.path.join(raw_data_dir, f))]
    
    for album_name in tqdm(album_folders, desc="Đang trích xuất ngữ nghĩa (CLIP)"):
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
                    inputs = processor(images=img, return_tensors="pt").to(device)
                    
                    # 1. Trích xuất đặc trưng
                    outputs = model.get_image_features(**inputs)
                    
                    # 2. Bóc vỏ Object để lấy Tensor (Xử lý lỗi BaseModelOutputWithPooling)
                    if not isinstance(outputs, torch.Tensor):
                        if hasattr(outputs, 'image_embeds'):
                            outputs = outputs.image_embeds
                        elif hasattr(outputs, 'pooler_output'):
                            outputs = outputs.pooler_output
                    
                    # 3. Sử dụng F.normalize an toàn của PyTorch
                    feature_vector = F.normalize(outputs, p=2, dim=-1)
                    feature_vector = feature_vector.squeeze(0).cpu()
                    
                    album_features.append(feature_vector)
                except Exception as e:
                    print(f"Lỗi khi xử lý ảnh {img_path}: {e}")
        
        if album_features:
            album_tensor = torch.stack(album_features)
            save_path = os.path.join(output_feature_dir, f"{album_name}.pt")
            torch.save(album_tensor, save_path)

# ... (Phần if __name__ == "__main__": giữ nguyên) ...

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RAW_DATA_DIRECTORY = os.path.join(BASE_DIR, "data", "raw_albums")
    OUTPUT_FEATURE_DIRECTORY = os.path.join(BASE_DIR, "data", "features")
    
    print("Bắt đầu trích xuất đặc trưng với cấu trúc Semantic Set...")
    extract_and_save_features(RAW_DATA_DIRECTORY, OUTPUT_FEATURE_DIRECTORY)