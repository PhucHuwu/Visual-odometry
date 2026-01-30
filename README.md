# Visual Odometry System

Hệ thống Visual Odometry (VO) sử dụng camera đơn (Monocular VO) để ước lượng và tái tạo quỹ đạo chuyển động của camera trong không gian 3D.

## Tính Năng

- **Nhiều thuật toán feature detection:**
    - FAST (Features from Accelerated Segment Test)
    - ORB (Oriented FAST and Rotated BRIEF)
    - SIFT (Scale-Invariant Feature Transform)
    - Lucas-Kanade Optical Flow

- **Hỗ trợ nhiều nguồn input:**
    - Webcam (live camera)
    - Video file (.mp4, .avi, etc.)

- **Motion estimation:**
    - Essential Matrix computation
    - Camera pose recovery (Rotation + Translation)
    - Monocular scale estimation

- **Output:**
    - 3D trajectory file
    - Runtime statistics (FPS, keypoints, matches)

## Cài Đặt

### 1. Tạo Conda Environment

```bash
conda env create -f environment.yml
conda activate vo
```

### 2. Cài Đặt Package

```bash
pip install -e .
```

## Sử Dụng

### Chạy với Webcam

```bash
python src/main.py --camera 0 --algorithm orb
```

### Chạy với Video File

```bash
python src/main.py --video data/sample_video.mp4 --algorithm sift
```

### Tùy Chọn Dòng Lệnh

```
--video PATH          Path tới video file (không dùng = webcam)
--camera ID           Camera device ID (default: 0)
--algorithm TYPE      Algorithm: fast, orb, sift, lucas_kanade (default: orb)
--config PATH         Path tới config file (default: config/default_config.yaml)
--output PATH         Output trajectory file (default: output/trajectory.txt)
--log-level LEVEL     Log level: DEBUG, INFO, WARNING, ERROR (default: INFO)
```

## Cấu Hình

Tất cả các file cấu hình nằm trong `config/`:

### `default_config.yaml`

- Video settings (resolution, FPS)
- Algorithm selection
- Visualization options
- Performance tuning
- Logging configuration

### `algorithm_config.yaml`

- Parameters cho từng algorithm (FAST, ORB, SIFT, LK)
- Feature matching settings (ratio test, RANSAC)

### `camera_params.yaml`

- Camera calibration parameters
- Intrinsic matrix (fx, fy, cx, cy)
- Distortion coefficients (k1, k2, k3, p1, p2)

## Camera Calibration

Trước khi sử dụng VO, cần calibrate camera:

1. Chụp 10-20 ảnh checkerboard pattern (9x6 inner corners)
2. Lưu vào `data/calibration_images/`
3. Chạy calibration script:

```bash
python scripts/calibrate_camera.py
```

4. File `config/camera_params.yaml` sẽ được tạo/cập nhật

## Cấu Trúc Thư Mục

```
Visual-odometry/
├── config/               # Configuration files
├── data/                 # Sample videos và calibration images
├── logs/                 # Log files
├── output/               # Trajectory output
├── src/                  # Source code
│   ├── core/            # Core VO components
│   ├── algorithms/      # Feature detection algorithms
│   ├── motion/          # Motion estimation
│   ├── utils/           # Utilities
│   └── main.py          # Entry point
└── tests/               # Unit tests
```

## Thuật Toán So Sánh

| Algorithm    | FPS (CPU) | Accuracy | Robustness | Use Case                   |
| ------------ | --------- | -------- | ---------- | -------------------------- |
| FAST         | 15-20     | Medium   | Low        | Real-time, smooth motion   |
| ORB          | 10-15     | Good     | Medium     | Balanced speed/accuracy    |
| SIFT         | 6-10      | Best     | High       | High accuracy, slow motion |
| Lucas-Kanade | 20-25     | Good     | Medium     | Smooth tracking, no jumps  |

## Giới Hạn

- **Monocular scale ambiguity:** Trajectory shape đúng nhưng scale có thể sai
- **Drift accumulation:** Lỗi tích lũy theo thời gian
- **Pure rotation:** VO thất bại khi camera chỉ rotate mà không translate
- **Low texture:** Ít keypoints trong scenes đơn giản (tường trắng, etc.)

## Development

### Run Tests

```bash
pytest tests/ -v --cov=src
```

### Code Style

```bash
pylint src/
```

## License

MIT License

## Contributors

Visual Odometry Team
