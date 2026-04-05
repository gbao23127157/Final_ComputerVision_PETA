"""
pec_utils.py
Các tiện ích đặc thù cho dataset PEC (Photo Event Classification):
  - build_labels_file   : tạo labels.txt từ cấu trúc thư mục images/
  - load_labels_dict    : đọc labels.txt → dict {album_id: label}
  - load_split          : đọc train.txt / test.txt → set album_ids
  - SubsetAlbumDataset  : Dataset con theo danh sách album_ids cụ thể
"""

import os
import json
import torch
import warnings
from torch.utils.data import Dataset


# ─────────────────────────────────────────────
# 1. Xây dựng file nhãn từ cấu trúc thư mục
# ─────────────────────────────────────────────

def build_labels_file(images_dir, output_label_file):
    """
    Duyệt qua images/<class_name>/<album_id>/ và tạo file nhãn.

    Tham số:
        images_dir (str)       : Đường dẫn đến thư mục images/ của PEC.
        output_label_file (str): Đường dẫn file đầu ra (labels.txt).

    Kết quả trả về:
        dict: class_to_idx — ánh xạ tên class → chỉ số nguyên.
    """
    class_names = sorted([
        d for d in os.listdir(images_dir)
        if os.path.isdir(os.path.join(images_dir, d))
    ])
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

    lines = []
    for cls_name in class_names:
        cls_dir = os.path.join(images_dir, cls_name)
        for album_id in os.listdir(cls_dir):
            if os.path.isdir(os.path.join(cls_dir, album_id)):
                lines.append(f"{album_id} {class_to_idx[cls_name]}")

    os.makedirs(os.path.dirname(output_label_file) or ".", exist_ok=True)
    with open(output_label_file, "w") as f:
        f.write("\n".join(lines))

    print(f"Đã tạo labels.txt: {len(lines)} album, {len(class_names)} class")
    return class_to_idx


# ─────────────────────────────────────────────
# 2. Đọc file nhãn
# ─────────────────────────────────────────────

def load_labels_dict(label_file):
    """
    Đọc labels.txt (mỗi dòng: <album_id> <label>) → dict.

    Tham số:
        label_file (str): Đường dẫn đến labels.txt.

    Kết quả trả về:
        dict: {album_id (str): label (int)}
    """
    labels = {}
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                labels[parts[0]] = int(parts[1])
    print(f"Đã tải {len(labels)} nhãn từ {label_file}")
    return labels


# ─────────────────────────────────────────────
# 3. Đọc split chính thức của PEC
# ─────────────────────────────────────────────

def load_split(meta_dir, split="train"):
    """
    Đọc file split của PEC và trả về tập album_id.

    Hỗ trợ 2 định dạng:
      - <split>.txt  : mỗi dòng là  class_name/album_id
      - <split>.json : {class_name: {album_id: [[img_id, ts], ...], ...}}

    Tham số:
        meta_dir (str): Đường dẫn thư mục meta/ của PEC.
        split    (str): "train" hoặc "test".

    Kết quả trả về:
        set: Tập hợp các album_id (str, chỉ phần số).
    """
    txt_file  = os.path.join(meta_dir, f"{split}.txt")
    json_file = os.path.join(meta_dir, f"{split}.json")
    album_ids = set()

    if os.path.exists(txt_file):
        with open(txt_file) as f:
            for line in f:
                line = line.strip()
                if "/" in line:
                    album_ids.add(line.split("/")[-1])
        print(f"Đọc {split}.txt: {len(album_ids)} album")

    elif os.path.exists(json_file):
        with open(json_file) as f:
            data = json.load(f)
        for albums in data.values():
            for aid in albums.keys():
                album_ids.add(str(aid))
        print(f"Đọc {split}.json: {len(album_ids)} album")

    else:
        raise FileNotFoundError(
            f"Không tìm thấy {split}.txt hoặc {split}.json trong {meta_dir}"
        )

    return album_ids


# ─────────────────────────────────────────────
# 4. Dataset theo danh sách album cụ thể
# ─────────────────────────────────────────────

class SubsetAlbumDataset(Dataset):
    """
    Dataset chỉ gồm các album thuộc một split cụ thể (train hoặc test).

    Tham số:
        feature_dir (str)  : Thư mục chứa các file <album_id>.pt.
        labels_dict (dict) : {album_id: label} từ load_labels_dict().
        album_ids   (set)  : Tập album_id cần đưa vào dataset.
    """

    def __init__(self, feature_dir, labels_dict, album_ids):
        self.feature_dir = feature_dir
        self.labels_dict = labels_dict
        self.album_files = [
            f"{aid}.pt"
            for aid in album_ids
            if (
                os.path.exists(os.path.join(feature_dir, f"{aid}.pt"))
                and aid in labels_dict
            )
        ]
        print(f"  SubsetAlbumDataset: {len(self.album_files)} album hợp lệ")

    def __len__(self):
        return len(self.album_files)

    def __getitem__(self, idx):
        fname    = self.album_files[idx]
        features = torch.load(
            os.path.join(self.feature_dir, fname), weights_only=True
        )
        label = self.labels_dict[fname[:-3]]
        return features, label
