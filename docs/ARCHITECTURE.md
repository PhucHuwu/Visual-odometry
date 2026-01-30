# Visual Odometry System Architecture

## 📐 Tổng Quan Kiến trúc

Hệ thống Visual Odometry được thiết kế theo kiến trúc modular với các layers:

```
┌──────────────────────────────────────────────────────┐
│                    Main Entry Point                   │
│                     (main.py)                        │
└──────────────────┬───────────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────────┐
│              VO Pipeline Orchestrator                │
│               (vo_pipeline.py)                       │
└──────┬──────────┬──────────┬──────────┬──────────────┘
       │          │          │          │
┌──────▼──┐  ┌───▼────┐ ┌───▼────┐ ┌───▼────────┐
│ Camera  │  │Preproc │ │Algorithms│ │  Motion    │
│ Input   │  │        │ │          │ │ Estimation │
└─────────┘  └────────┘ └────┬─────┘ └─────┬──────┘
                             │              │
              ┌──────────────▼──────────────▼──────┐
              │        Trajectory Manager          │
              │         (trajectory.py)            │
              └────────────────────────────────────┘
```

## 🧩 Components Chi Tiết

### 1. Core Layer

#### `camera.py` - Camera Input Handler

- **Chức năng:** Xử lý input từ webcam hoặc video file
- **Features:**
    - Support cả live camera và video file
    - Auto resize frame
    - Progress tracking cho video file
    - Context manager support (with statement)

#### `preprocessor.py` - Image Preprocessing

- **Chức năng:** Tiền xử lý ảnh
- **Pipeline:**
    1. BGR → Grayscale conversion
    2. Optional: Gaussian blur để denoise

#### `trajectory.py` - Trajectory Management

- **Chức năng:** Tích lũy và quản lý camera pose history
- **Algorithm:**
    ```
    T_world = T_world * T_current
    position += rotation_world @ translation_current
    ```
- **Features:**
    - History limit (tránh memory overflow)
    - Save/load trajectory

#### `vo_pipeline.py` - Main Pipeline Orchestrator

- **Chức năng:** Điều phối toàn bộ VO process
- **Flow:**
    ```
    Frame → Undistort → Preprocess → Feature Detection/Tracking
      → Essential Matrix → Pose Recovery → Trajectory Update
    ```

### 2. Algorithms Layer

#### Base Architecture

```python
BaseAlgorithm (ABC)
    ├── detect(image) → (keypoints, descriptors)
    ├── match(kp1, desc1, kp2, desc2) → matches
    └── extract_matched_points(kp1, kp2, matches) → (pts1, pts2)
```

#### Implementations

1. **FAST Detector** (`fast_detector.py`)
    - Detection: FAST corners
    - Description: ORB descriptors
    - Matching: BFMatcher với Hamming distance
    - Pros: Nhanh nhất (15-20 FPS)
    - Cons: Ít robust với scale/rotation changes

2. **ORB Detector** (`orb_detector.py`)
    - Detection: Oriented FAST
    - Description: Rotated BRIEF (256-bit binary)
    - Matching: BFMatcher với Hamming distance
    - Pros: Balance tốt speed/accuracy
    - Cons: Limited scale invariance

3. **SIFT Detector** (`sift_detector.py`)
    - Detection: DoG (Difference of Gaussians)
    - Description: 128-dim float descriptors
    - Matching: BFMatcher với L2 distance
    - Pros: Scale/rotation invariant, robust nhất
    - Cons: Chậm nhất (6-10 FPS)

4. **Lucas-Kanade Optical Flow** (`lk_optical_flow.py`)
    - Tracking: Sparse optical flow
    - Initial detection: FAST corners
    - Matching: N/A (tracking-based)
    - Pros: Smooth tracking, nhanh (20-25 FPS)
    - Cons: Không handle large motion, keypoint drift

### 3. Motion Estimation Layer

#### `essential_matrix.py`

- **Algorithm:** 5-point algorithm với RANSAC
- **Input:** Matched points + camera matrix K
- **Output:** Essential Matrix E + inlier mask
- **Validation:** Kiểm tra rank(E) = 2

#### `pose_estimation.py`

- **Algorithm:** Decompose E thành 4 solutions, chọn valid solution
- **Input:** E + matched points + K
- **Output:** R (3x3 rotation) + t (3x1 translation, unit vector)
- **Validation:**
    - det(R) = 1
    - R^T \* R = I

#### `scale_estimation.py`

- **Problem:** Monocular VO có scale ambiguity
- **Methods:**
    - Constant velocity assumption
    - Unit scale (default)
    - Median depth heuristic
- **Limitation:** Absolute scale không xác định được

### 4. Utilities Layer

#### `config_loader.py`

- Load YAML configs
- Nested key access: `config.get("camera.fx")`
- Runtime override support

#### `logger.py`

- File + console logging
- Log rotation (10MB per file)
- Configurable levels

#### `calibration.py`

- Camera calibration từ checkerboard
- Undistortion
- Load/save calibration parameters

## 📊 Data Flow

### Normal Operation (Feature Matching)

```
Frame N-1                           Frame N
    │                                  │
    ├──► Detect Features              │
    │    (FAST/ORB/SIFT)               │
    │        │                          │
    │        └──────────────┬───────────┘
    │                       │
    │                   Match Features
    │                   (BFMatcher + Ratio Test)
    │                       │
    │                       ▼
    │                Matched Points (pts1, pts2)
    │                       │
    │                       ▼
    │             Estimate Essential Matrix E
    │             (RANSAC, inlier filtering)
    │                       │
    │                       ▼
    │                 Recover Pose (R, t)
    │                       │
    │                       ▼
    │                 Scale Estimation
    │                       │
    │                       ▼
    └───────────► Update Trajectory
```

### Optical Flow Tracking

```
Frame N-1                           Frame N
    │                                  │
    ├──► Detect Keypoints             │
    │    (FAST)                        │
    │        │                          │
    │        └──────────────┬───────────┘
    │                       │
    │              Track Keypoints
    │              (calcOpticalFlowPyrLK)
    │                       │
    │                       ▼
    │                Tracked Points (pts1, pts2)
    │                       │
    │             (Tương tự như trên)
    │                       │
    └─────────────────────────►
```

## ⚙️ Configuration Hierarchy

```
default_config.yaml
    ├── video: source, resolution, FPS
    ├── algorithm: type selection
    ├── camera: calibration file path
    ├── visualization: display options
    ├── performance: threading, profiling
    └── logging: level, file, rotation

algorithm_config.yaml
    ├── fast: threshold, nonmaxSuppression
    ├── orb: nfeatures, scaleFactor, nlevels
    ├── sift: nfeatures, contrastThreshold
    ├── lucas_kanade: winSize, maxLevel
    └── matching: ratio_test, ransac_threshold

camera_params.yaml
    ├── camera_matrix: fx, fy, cx, cy
    ├── distortion_coefficients: k1, k2, k3, p1, p2
    └── image_size: width, height
```

## 🔄 State Management

### VO Pipeline State

```python
prev_image: np.ndarray           # Previous frame (grayscale)
prev_keypoints: List[KeyPoint]   # Previous keypoints
prev_descriptors: np.ndarray     # Previous descriptors
frame_count: int                 # Total frames processed
```

### Trajectory State

```python
positions: List[np.ndarray]      # History of 3D positions
orientations: List[np.ndarray]   # History of rotation matrices
current_position: np.ndarray     # Current camera position
current_rotation: np.ndarray     # Current camera orientation
```

## 🚨 Error Handling

### Pipeline Failures

1. **Không đủ keypoints:**
    - Skip frame, continue với frame tiếp theo
    - Log warning

2. **Essential Matrix estimation thất bại:**
    - Skip frame
    - Không update trajectory

3. **Pose recovery thất bại:**
    - Kiểm tra rotation matrix validity
    - Skip nếu invalid

### Recovery Strategies

- **Keypoint depletion:** Re-detect keypoints (LK tracking)
- **Low inlier ratio:** Increase RANSAC threshold (runtime)
- **Drift accumulation:** Log drift metrics, document limitation

## 📈 Performance Optimization

### CPU Optimization (Implemented)

1. **NumPy vectorization:** Matrix operations
2. **Multi-threading potential:**
    - Thread 1: Frame capture
    - Thread 2: Feature detection
    - Thread 3: Visualization (future)

### Future Optimizations

1. **GPU acceleration:** OpenCV CUDA modules
2. **Keypoint reduction:** Adaptive nfeatures based on FPS
3. **Frame skipping:** Process every Nth frame
4. **Resolution downscaling:** Trade accuracy for speed

## 🔍 Testing Strategy

### Unit Tests

- `test_algorithms.py`: Test từng algorithm
- `test_motion_estimation.py`: Test E, R, t estimation
- `test_preprocessing.py`: Test grayscale, denoise

### Integration Tests

- `test_integration.py`: End-to-end pipeline test
- Verify trajectory shape với known motion

### Performance Benchmarks

- `benchmark.py`: FPS cho từng algorithm
- Compare CPU vs theoretical limits

## 📚 References

### Algorithms

- FAST: [Rosten & Drummond, 2006]
- ORB: [Rublee et al., 2011]
- SIFT: [Lowe, 2004]
- Lucas-Kanade: [Lucas & Kanade, 1981]

### Visual Odometry

- [Scaramuzza & Fraundorfer, 2011] - Visual Odometry Tutorial
- [Nistér, 2004] - 5-point algorithm cho E matrix
- [Hartley & Zisserman, 2004] - Multiple View Geometry

---

**Version:** 0.1.0  
**Last Updated:** 2026-01-30
