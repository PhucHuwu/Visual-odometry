# Tài Liệu Yêu Cầu: Visual Odometry (Định Vị Thị Giác)

## 1. Tổng Quan Dự Án

Xây dựng một chương trình phần mềm sử dụng kỹ thuật Visual Odometry (VO) trên camera đơn (Monocular VO) để ước lượng và tái tạo quỹ đạo chuyển động của camera trong không gian.

## 2. Yêu Cầu Hệ Thống

### 2.1. Đầu Vào (Input)

Hệ thống cần hỗ trợ hai nguồn dữ liệu đầu vào:

- **Camera trực tiếp (Live Feed):** Stream video thời gian thực từ webcam laptop hoặc camera rời kết nối qua USB.
- **Video có sẵn (Offline):** File video đã được ghi lại từ trước (định dạng phổ biến như .mp4, .avi).

### 2.2. Chức Năng Xử Lý (Core Processing)

Chương trình thực hiện các bước xử lý ảnh để tính toán chuyển động:

1.  **Thu nhận ảnh:** Đọc frame từ nguồn đầu vào.
2.  **Tiền xử lý:** Chuyển đổi thang xám (grayscale), khử nhiễu (nếu cần).
3.  **Trích xuất đặc trưng (Feature Extraction):** Phát hiện các điểm đặc trưng (keypoints) trong ảnh (sử dụng các thuật toán như FAST, ORB, SIFT, hoặc Lucas-Kanade Optical Flow).
4.  **Ước lượng chuyển động (Motion Estimation):** Tính toán Ma trận thiết yếu (Essential Matrix) hoặc Ma trận cơ bản (Fundamental Matrix) giữa các frame liên tiếp để tìm ra hướng di chuyển (Rotation - R) và tịnh tiến (Translation - t).
5.  **Tái tạo quỹ đạo:** Cập nhật vị trí hiện tại của camera dựa trên R và t.

### 2.3. Đầu Ra (Output)

- **Hiển thị trực quan:** Vẽ quỹ đạo chuyển động của camera lên màn hình.
- **Chế độ hiển thị:**
    - **2D:** Bản đồ quỹ đạo trên mặt phẳng (ví dụ: nhìn từ trên xuống - Bird's eye view).
    - **3D:** (Tùy chọn nâng cao) Quỹ đạo trong không gian 3 chiều.
- **Thông tin bổ sung:** Hiển thị video gốc song song với bản đồ quỹ đạo để đối chiếu.

## 3. Yêu Cầu Phi Chức Năng

- **Hiệu năng:** Có khả năng chạy mượt mà (real-time hoặc near real-time) trên cấu hình laptop thông thường.
- **Độ chính xác:** Đảm bảo ước lượng quỹ đạo tương đối chính xác, hạn chế sai số tích lũy (drift) trong thời gian ngắn.
