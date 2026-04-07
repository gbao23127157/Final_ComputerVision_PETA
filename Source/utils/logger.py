import logging
import os
import sys

def setup_logger(log_file_path="training_log.txt"):
    """
    Thiết lập công cụ ghi nhật ký (logger).
    Logger này sẽ in thông báo ra màn hình console (để theo dõi trực tiếp)
    đồng thời ghi vào một file văn bản (để lưu trữ làm minh chứng cho báo cáo).
    
    Tham số:
        log_file_path (str): Đường dẫn đến file lưu nhật ký.
        
    Kết quả trả về:
        logging.Logger: Đối tượng logger để gọi các hàm .info(), .error().
    """
    # Lấy đối tượng logger mặc định
    logger = logging.getLogger("PETA_Project")
    
    # Thiết lập mức độ ghi log thấp nhất là INFO
    logger.setLevel(logging.INFO)
    
    # Tránh việc logger ghi đúp dòng nếu được gọi nhiều lần
    if logger.handlers:
        return logger

    # Định dạng chuỗi thông báo: [Thời gian] - [Mức độ] - Nội dung
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # 1. Handler ghi ra màn hình console (Terminal)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Đảm bảo thư mục chứa file log tồn tại
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 2. Handler ghi vào file văn bản
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger