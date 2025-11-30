# Heatmap và Annotated Images

## Cần thêm 2 ảnh vào thư mục này:

### 1. heatmap.png
- Ảnh X-quang với heatmap overlay (màu đỏ-vàng-xanh)
- Hiển thị vùng AI phát hiện bất thường
- Màu đỏ: Bất thường cao
- Màu vàng: Nghi ngờ
- Màu xanh: Bình thường

### 2. annotated.png
- Ảnh X-quang với bounding box chính xác
- Có khoanh vùng các tổn thương thực tế
- Có label cho từng vùng (Đám mờ phổi, Xơ hóa, etc.)

## Cách tạo ảnh mock tạm thời:
1. Copy file `origin.png` thành `heatmap.png`
2. Copy file `origin.png` thành `annotated.png`

Sau này có thể thay thế bằng ảnh thật với heatmap và annotation.
