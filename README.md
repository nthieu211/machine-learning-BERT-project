# MACHINE LEARNING BERT PROJECT 

# Thông tin môn học
- Nhập môn học máy

- Giáo viên hướng dẫn: TS. Bùi Tiến Lên


# Thông tin nhóm
|MSSV|Họ và Tên|
|---|---|
|20120240|Dương Thị An|
|20120246|Nguyễn Hoàng Anh|
|20120270|Cao Tấn Đức|
|20120284|Lê Đức Hậu|
|20120288|Nguyễn Trung Hiếu|


# Nội dung bài làm 
- Xây dựng mô hình BERT trả lời các câu hỏi trắc nghiệm điền từ trong TOEIC dựa trên bộ dữ liệu khoảng 3600 câu hỏi - câu trả lời.
- Triển khai ứng dụng web để sử dụng mô hình BERT đã xây dựng.


# Cấu trúc thư mục:
```
├── README.md
├── app.py
├── model_training
│   ├── data
│   │   ├── data.json
│   │   ├── dataset_type_1.json
│   │   ├── dataset_type_2.json
│   │   ├── dataset_type_3.json
│   │   ├── dataset_type_4.json
│   │   └── dataset_type_5.json
│   ├── main.py
│   ├── model_pytorch.pt
│   └── requirements.txt
├── requirements.txt
├── static
│   └── favicon.ico
└── templates
    └── index.html
```


# Để thực hiện huấn luyện lại mô hình, cần thực hiện bước:
1. Cài đặt các thư viện cần thiết trong `model_training\requirements.txt`.
2. Chạy file python `main.py`.
3. Kết quả: mô hình mới sẽ được tạo ra và lưu thành file `model_pytorch.pt`.


# Để chạy ứng dụng web, cần thực hiện bước:
1. Cài đặt các thư viện cần thiết trong `requirements.txt`.
2. Chạy file `app.py` để khởi chạy localhost.
3. Mở trang localhost mà ứng dụng đang chạy để sử dụng.
