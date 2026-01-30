# Visual Odometry System - Project Plan

## 📋 Overview

**Mục tiêu:** Xây dựng hệ thống Visual Odometry (VO) production-ready sử dụng camera đơn (Monocular VO) để ước lượng và tái tạo quỹ đạo chuyển động của camera trong không gian 3D. Hệ thống hỗ trợ nhiều thuật toán feature extraction và cho phép người dùng chọn lựa để so sánh hiệu năng.

**Tại sao:** Visual Odometry là nền tảng cho nhiều ứng dụng như robot navigation, autonomous vehicles, AR/VR. Dự án này tạo ra công cụ linh hoạt để nghiên cứu và so sánh các thuật toán VO khác nhau.

---

## 🎯 Project Type

**BACKEND/STANDALONE APPLICATION** (Computer Vision Application)

- Không phải Web App → Không dùng `frontend-specialist`
- Không phải Mobile App → Không dùng `mobile-developer`
- Ứng dụng desktop độc lập với GUI → Dùng `backend-specialist`, `performance-optimizer`

---

## ✅ Success Criteria

| Tiêu Chí           | Định Nghĩa Thành Công                                                             |
| ------------------ | --------------------------------------------------------------------------------- |
| **Chức năng**      | Hệ thống chạy được với cả live camera và video file, tái tạo quỹ đạo 3D chính xác |
| **Thuật toán**     | Hỗ trợ đầy đủ 4 thuật toán: FAST, ORB, SIFT, Lucas-Kanade Optical Flow            |
| **Hiệu năng**      | Đạt ≥10 FPS với CPU (Intel i5/M1 hoặc tương đương) với video 720p                 |
| **Độ chính xác**   | Drift error < 5% trên đoạn video 30 giây (so với ground truth nếu có)             |
| **Cross-platform** | Chạy mượt trên cả macOS (M1/Intel) và Windows 10/11                               |
| **UX**             | GUI rõ ràng, dễ chọn thuật toán, hiển thị real-time trajectory                    |
| **Production**     | Error handling đầy đủ, logging, camera calibration, config file                   |

---

## 🛠️ Tech Stack

| Component              | Technology                           | Rationale                                           |
| ---------------------- | ------------------------------------ | --------------------------------------------------- |
| **Environment**        | Conda environment "vo"               | Isolated dependencies, reproducible setup           |
| **Core Language**      | Python 3.10+                         | Ecosystem mạnh cho CV, OpenCV native support        |
| **Computer Vision**    | OpenCV 4.8+                          | Feature extraction, camera matrix, essential matrix |
| **Numerical**          | NumPy 1.24+                          | Matrix operations, linear algebra                   |
| **3D Visualization**   | Open3D hoặc Matplotlib 3D            | Real-time 3D trajectory plotting                    |
| **GUI Framework**      | PyQt5 hoặc Tkinter                   | Cross-platform, algorithm selection, video controls |
| **CPU Optimization**   | NumPy vectorization, multi-threading | Tối ưu CPU cho feature detection/matching           |
| **Camera Calibration** | OpenCV calibration module            | Intrinsic/extrinsic parameters                      |
| **Video I/O**          | OpenCV VideoCapture                  | USB camera và video file support                    |
| **Config Management**  | YAML/JSON                            | Algorithm parameters, camera settings               |
| **Testing**            | pytest                               | Unit tests cho từng module                          |
| **Logging**            | Python logging                       | Debug và performance monitoring                     |

**Trade-offs:**

- **Open3D vs Matplotlib 3D:** Open3D mạnh hơn cho point cloud nhưng dependency lớn hơn. Matplotlib nhẹ hơn, đủ cho trajectory.
- **PyQt5 vs Tkinter:** PyQt5 đẹp hơn, nhiều widget hơn nhưng cần license cho commercial. Tkinter built-in, đơn giản hơn.
- **CPU-only:** Đơn giản setup, cross-platform tốt hơn, nhưng FPS thấp hơn GPU (10-12 FPS vs 25-30 FPS).

---

## 📁 File Structure

```
Visual-odometry/
├── README.md
├── environment.yml               # Conda environment definition
├── requirements.txt              # Pip requirements (backup)
├── setup.py
├── config/
│   ├── camera_params.yaml        # Camera calibration parameters
│   ├── algorithm_config.yaml     # Algorithm-specific settings
│   └── default_config.yaml       # Default runtime config
├── src/
│   ├── __init__.py
│   ├── main.py                   # Entry point
│   ├── core/
│   │   ├── __init__.py
│   │   ├── camera.py             # Camera input handler (live/file)
│   │   ├── preprocessor.py       # Grayscale, denoise
│   │   ├── vo_pipeline.py        # Main VO orchestrator
│   │   └── trajectory.py         # Trajectory accumulation
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── base_algorithm.py     # Abstract base class
│   │   ├── fast_detector.py      # FAST feature detection
│   │   ├── orb_detector.py       # ORB feature detection
│   │   ├── sift_detector.py      # SIFT feature detection
│   │   └── lk_optical_flow.py    # Lucas-Kanade Optical Flow
│   ├── motion/
│   │   ├── __init__.py
│   │   ├── essential_matrix.py   # Essential matrix estimation
│   │   ├── pose_estimation.py    # Recover R, t from E
│   │   └── scale_estimation.py   # Monocular scale ambiguity handling
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── trajectory_3d.py      # 3D trajectory plotter
│   │   ├── frame_display.py      # Video frame + keypoints overlay
│   │   └── stats_panel.py        # FPS, drift, algorithm info
│   ├── gui/
│   │   ├── __init__.py
│   │   ├── main_window.py        # Main GUI window
│   │   ├── algorithm_selector.py # Algorithm selection widget
│   │   └── video_controls.py     # Play/pause/speed controls
│   └── utils/
│       ├── __init__.py
│       ├── calibration.py        # Camera calibration utilities
│       ├── logger.py             # Logging setup
│       └── config_loader.py      # YAML config parser
├── tests/
│   ├── __init__.py
│   ├── test_algorithms.py
│   ├── test_motion_estimation.py
│   ├── test_preprocessing.py
│   └── test_integration.py
├── data/
│   ├── sample_videos/            # Test videos
│   └── calibration_images/       # Camera calibration images
├── docs/
│   ├── ARCHITECTURE.md           # System architecture
│   ├── ALGORITHMS.md             # Algorithm comparison
│   └── CALIBRATION.md            # How to calibrate camera
└── scripts/
    ├── calibrate_camera.py       # Camera calibration script
    └── benchmark.py              # Performance benchmarking
```

---

## 📊 Task Breakdown

### **P0: Foundation Setup** (Dependency: None)

#### Task 1.1: Project Initialization

- **Agent:** `backend-specialist`
- **Skill:** `python-patterns`, `clean-code`
- **Priority:** P0
- **Dependencies:** None
- **INPUT:** Requirements, tech stack
- **OUTPUT:**
    - `environment.yml` cho conda environment "vo"
    - `requirements.txt` với pinned versions (backup)
    - `setup.py` for package installation
    - Folder structure theo design
    - `.gitignore` (data/, \*.pyc, **pycache**, .conda/)
- **VERIFY:**
    - `conda env create -f environment.yml` tạo environment thành công
    - `conda activate vo` kích hoạt environment
    - `pip install -e .` chạy thành công, import src modules không lỗi

#### Task 1.2: Configuration System

- **Agent:** `backend-specialist`
- **Skill:** `clean-code`
- **Priority:** P0
- **Dependencies:** Task 1.1
- **INPUT:** Config requirements (camera params, algorithm settings)
- **OUTPUT:**
    - `config_loader.py` với YAML parsing
    - `default_config.yaml`, `camera_params.yaml`, `algorithm_config.yaml`
- **VERIFY:** Load config thành công, override parameters hoạt động

#### Task 1.3: Logging System

- **Agent:** `backend-specialist`
- **Skill:** `clean-code`
- **Priority:** P0
- **Dependencies:** Task 1.1
- **INPUT:** Logging requirements (debug, info, error levels)
- **OUTPUT:**
    - `logger.py` với file và console handlers
    - Log rotation setup
- **VERIFY:** Log messages xuất hiện đúng format, file logs được tạo

---

### **P1: Core Camera Input** (Dependency: P0)

#### Task 2.1: Camera Input Handler

- **Agent:** `backend-specialist`
- **Skill:** `python-patterns`
- **Priority:** P1
- **Dependencies:** Task 1.1, 1.2, 1.3
- **INPUT:** Live camera và video file requirements
- **OUTPUT:**
    - `camera.py` với class `CameraInput`
    - Support `VideoCapture` cho cả webcam (device ID) và file path
    - Frame buffering nếu cần
- **VERIFY:**
    - Mở webcam thành công, đọc được frames
    - Mở video file thành công, đọc được frames
    - FPS tracking chính xác

#### Task 2.2: Preprocessing Pipeline

- **Agent:** `backend-specialist`
- **Skill:** `clean-code`
- **Priority:** P1
- **Dependencies:** Task 2.1
- **INPUT:** Raw BGR frames từ camera
- **OUTPUT:**
    - `preprocessor.py` với grayscale conversion
    - Optional: Gaussian blur để denoise
    - Resize nếu cần (maintain aspect ratio)
- **VERIFY:** Output frame shape đúng, grayscale conversion chính xác

---

### **P2: Algorithm Implementation** (Dependency: P1)

#### Task 3.1: Base Algorithm Interface

- **Agent:** `backend-specialist`
- **Skill:** `python-patterns`, `clean-code`
- **Priority:** P2
- **Dependencies:** Task 2.2
- **INPUT:** Algorithm requirements (detect, describe, match)
- **OUTPUT:**
    - `base_algorithm.py` với Abstract Base Class
    - Methods: `detect()`, `describe()`, `match()`
    - Common interface cho tất cả algorithms
- **VERIFY:** Subclass có thể inherit và override methods

#### Task 3.2: FAST Feature Detector

- **Agent:** `backend-specialist`
- **Skill:** `clean-code`
- **Priority:** P2
- **Dependencies:** Task 3.1
- **INPUT:** Grayscale frame
- **OUTPUT:**
    - `fast_detector.py` implement FAST
    - Keypoints detection với configurable threshold
- **VERIFY:** Detect keypoints trên test image, visualize keypoints

#### Task 3.3: ORB Feature Detector

- **Agent:** `backend-specialist`
- **Skill:** `clean-code`
- **Priority:** P2
- **Dependencies:** Task 3.1
- **INPUT:** Grayscale frame
- **OUTPUT:**
    - `orb_detector.py` implement ORB
    - Keypoints + descriptors
- **VERIFY:** Detect và match keypoints giữa 2 frames

#### Task 3.4: SIFT Feature Detector

- **Agent:** `backend-specialist`
- **Skill:** `clean-code`
- **Priority:** P2
- **Dependencies:** Task 3.1
- **INPUT:** Grayscale frame
- **OUTPUT:**
    - `sift_detector.py` implement SIFT
    - Keypoints + descriptors (128-dim)
- **VERIFY:** SIFT detection hoạt động, match accuracy tốt hơn FAST/ORB

#### Task 3.5: Lucas-Kanade Optical Flow

- **Agent:** `backend-specialist`
- **Skill:** `clean-code`
- **Priority:** P2
- **Dependencies:** Task 3.1
- **INPUT:** 2 consecutive grayscale frames
- **OUTPUT:**
    - `lk_optical_flow.py` implement Lucas-Kanade
    - Track keypoints từ frame trước sang frame sau
- **VERIFY:** Tracking keypoints mượt mà, outlier rejection hoạt động

---

### **P3: Motion Estimation** (Dependency: P2)

#### Task 4.1: Essential Matrix Estimation

- **Agent:** `backend-specialist`
- **Skill:** `python-patterns`
- **Priority:** P3
- **Dependencies:** Task 3.2, 3.3, 3.4, 3.5
- **INPUT:** Matched keypoints từ 2 frames, camera intrinsic matrix K
- **OUTPUT:**
    - `essential_matrix.py` với `cv2.findEssentialMat()`
    - RANSAC outlier rejection
    - Inlier mask return
- **VERIFY:** Essential matrix có rank 2, satisfies E^T \* E = 0

#### Task 4.2: Pose Estimation (R, t Recovery)

- **Agent:** `backend-specialist`
- **Skill:** `clean-code`
- **Priority:** P3
- **Dependencies:** Task 4.1
- **INPUT:** Essential matrix E, matched points
- **OUTPUT:**
    - `pose_estimation.py` với `cv2.recoverPose()`
    - Extract Rotation R và Translation t
    - Handle 4 possible solutions → chọn đúng
- **VERIFY:** R là rotation matrix (det(R) = 1, R^T \* R = I), t unit vector

#### Task 4.3: Scale Estimation (Monocular)

- **Agent:** `backend-specialist`
- **Skill:** `python-patterns`
- **Priority:** P3
- **Dependencies:** Task 4.2
- **INPUT:** Translation vector t (ambiguous scale)
- **OUTPUT:**
    - `scale_estimation.py` với strategy:
        - Option 1: Assume constant velocity
        - Option 2: Use ground truth nếu có
        - Option 3: Heuristic (median depth estimation)
- **VERIFY:** Trajectory không explode hoặc collapse, reasonable scale

---

### **P4: VO Pipeline Integration** (Dependency: P3)

#### Task 5.1: Trajectory Accumulation

- **Agent:** `backend-specialist`
- **Skill:** `clean-code`
- **Priority:** P4
- **Dependencies:** Task 4.3
- **INPUT:** R, t từ mỗi frame pair
- **OUTPUT:**
    - `trajectory.py` với class `Trajectory`
    - Accumulate poses: `T_world = T_world * T_current`
    - Store 3D positions (x, y, z) history
- **VERIFY:** Trajectory array không có NaN, infinity

#### Task 5.2: Main VO Pipeline

- **Agent:** `backend-specialist`
- **Skill:** `clean-code`
- **Priority:** P4
- **Dependencies:** Task 5.1, all P2, P3 tasks
- **INPUT:** Video stream, selected algorithm
- **OUTPUT:**
    - `vo_pipeline.py` orchestrate:
        1. Read frame → Preprocess
        2. Feature detection/tracking
        3. Motion estimation
        4. Trajectory update
    - Loop cho toàn bộ video
- **VERIFY:** Pipeline chạy end-to-end, output trajectory hợp lý

---

### **P5: Visualization** (Dependency: P4)

#### Task 6.1: 3D Trajectory Visualization

- **Agent:** `backend-specialist`
- **Skill:** `clean-code`
- **Priority:** P5
- **Dependencies:** Task 5.2
- **INPUT:** Trajectory 3D positions
- **OUTPUT:**
    - `trajectory_3d.py` với Matplotlib 3D hoặc Open3D
    - Real-time update plot (every N frames)
    - Camera orientation visualization (optional)
- **VERIFY:** 3D plot hiển thị đúng, rotate/zoom hoạt động

#### Task 6.2: Frame Display with Keypoints

- **Agent:** `backend-specialist`
- **Skill:** `clean-code`
- **Priority:** P5
- **Dependencies:** Task 5.2
- **INPUT:** Current frame, detected keypoints
- **OUTPUT:**
    - `frame_display.py` overlay keypoints lên video
    - Matches visualization (lines giữa frames nếu dùng matching)
- **VERIFY:** Keypoints hiển thị rõ ràng trên video

#### Task 6.3: Stats Panel

- **Agent:** `backend-specialist`
- **Skill:** `clean-code`
- **Priority:** P5
- **Dependencies:** Task 5.2
- **INPUT:** Runtime stats (FPS, num keypoints, inliers, etc.)
- **OUTPUT:**
    - `stats_panel.py` hiển thị real-time metrics
    - Text overlay hoặc separate panel
- **VERIFY:** Stats update real-time, chính xác

---

### **P6: GUI Development** (Dependency: P5)

#### Task 7.1: Main Window

- **Agent:** `backend-specialist`
- **Skill:** `clean-code`
- **Priority:** P6
- **Dependencies:** Task 6.1, 6.2, 6.3
- **INPUT:** All visualization components
- **OUTPUT:**
    - `main_window.py` với PyQt5/Tkinter
    - Layout: Video frame bên trái, 3D plot bên phải, controls dưới
- **VERIFY:** Window hiển thị đúng layout, resize hoạt động

#### Task 7.2: Algorithm Selector

- **Agent:** `backend-specialist`
- **Skill:** `clean-code`
- **Priority:** P6
- **Dependencies:** Task 7.1
- **INPUT:** List algorithms (FAST, ORB, SIFT, LK)
- **OUTPUT:**
    - `algorithm_selector.py` với dropdown/radio buttons
    - Signal khi user đổi algorithm
- **VERIFY:** Chọn algorithm → pipeline restart với algorithm mới

#### Task 7.3: Video Controls

- **Agent:** `backend-specialist`
- **Skill:** `clean-code`
- **Priority:** P6
- **Dependencies:** Task 7.1
- **INPUT:** Video playback state
- **OUTPUT:**
    - `video_controls.py` với Play/Pause/Stop/Speed
    - File browser để chọn video
    - Camera selection dropdown
- **VERIFY:** Controls hoạt động, video pause/resume chính xác

---

### **P7: Camera Calibration** (Dependency: None, can parallel)

#### Task 8.1: Calibration Utilities

- **Agent:** `backend-specialist`
- **Skill:** `python-patterns`
- **Priority:** P7
- **Dependencies:** Task 1.1
- **INPUT:** Checkerboard calibration images
- **OUTPUT:**
    - `calibration.py` với calibration pipeline
    - Save/load camera matrix K và distortion coefficients
    - `scripts/calibrate_camera.py` standalone script
- **VERIFY:** Calibrate camera với checkerboard, K matrix hợp lý (fx, fy ~ focal length)

---

### **P8: Performance Optimization** (Dependency: P6)

#### Task 9.1: CPU Optimization

- **Agent:** `performance-optimizer`
- **Skill:** `performance-profiling`
- **Priority:** P8
- **Dependencies:** Task 5.2
- **INPUT:** Baseline VO pipeline
- **OUTPUT:**
    - NumPy vectorization cho matrix operations
    - Optimize feature detection parameters (reduce keypoints nếu cần)
    - Caching cho camera matrix, config
    - Profiling với cProfile để tìm bottlenecks
- **VERIFY:** FPS tăng ≥30% so với baseline

#### Task 9.2: Multi-threading

- **Agent:** `performance-optimizer`
- **Skill:** `performance-profiling`
- **Priority:** P8
- **Dependencies:** Task 9.1
- **INPUT:** Single-threaded pipeline
- **OUTPUT:**
    - Separate threads: frame capture, processing, visualization
    - Thread-safe queues
- **VERIFY:** FPS tăng thêm, không deadlock

---

### **P9: Testing** (Dependency: P8)

#### Task 10.1: Unit Tests

- **Agent:** `test-engineer`
- **Skill:** `testing-patterns`
- **Priority:** P9
- **Dependencies:** All implementation tasks
- **INPUT:** All modules
- **OUTPUT:**
    - `tests/test_algorithms.py` test từng algorithm
    - `tests/test_motion_estimation.py` test E, R, t
    - `tests/test_preprocessing.py` test preprocessing
    - Coverage ≥70%
- **VERIFY:** `pytest` pass tất cả tests

#### Task 10.2: Integration Tests

- **Agent:** `test-engineer`
- **Skill:** `testing-patterns`
- **Priority:** P9
- **Dependencies:** Task 10.1
- **INPUT:** Full pipeline
- **OUTPUT:**
    - `tests/test_integration.py` test end-to-end
    - Test với sample video, verify trajectory shape
- **VERIFY:** Integration test pass, trajectory không drift quá 10%

#### Task 10.3: Performance Benchmarks

- **Agent:** `performance-optimizer`
- **Skill:** `performance-profiling`
- **Priority:** P9
- **Dependencies:** Task 10.2
- **INPUT:** Sample videos (720p, 1080p)
- **OUTPUT:**
    - `scripts/benchmark.py` benchmark FPS cho từng algorithm
    - CSV report: Algorithm, Resolution, FPS, Memory, CPU Usage
    - Include multi-threading ON/OFF comparison
- **VERIFY:** FAST ≥15 FPS, ORB ≥10 FPS, SIFT ≥6 FPS trên 720p (CPU)

---

### **P10: Documentation** (Dependency: P9)

#### Task 11.1: Code Documentation

- **Agent:** `documentation-writer`
- **Skill:** `documentation-templates`
- **Priority:** P10
- **Dependencies:** All implementation
- **INPUT:** Source code
- **OUTPUT:**
    - Docstrings cho tất cả public functions/classes
    - Type hints (Python 3.10+)
- **VERIFY:** `pydoc` generate docs thành công

#### Task 11.2: User Documentation

- **Agent:** `documentation-writer`
- **Skill:** `documentation-templates`
- **Priority:** P10
- **Dependencies:** Task 11.1
- **INPUT:** System functionality
- **OUTPUT:**
    - `README.md`: Installation, Quick Start, Usage
    - `docs/ARCHITECTURE.md`: System design
    - `docs/ALGORITHMS.md`: Algorithm comparison table
    - `docs/CALIBRATION.md`: How to calibrate camera
- **VERIFY:** Follow README → có thể run app thành công

---

## 🔍 Phase X: Final Verification

> 🔴 **CRITICAL:** Tất cả checks này PHẢI pass trước khi đánh dấu project complete.

### 1. Functional Tests

```bash
# Activate conda environment first
conda activate vo

# Run all unit tests
pytest tests/ -v --cov=src --cov-report=html

# Expected: Coverage ≥70%, all tests pass
```

### 2. Integration Test

```bash
# Activate environment
conda activate vo

# Run với sample video
python src/main.py --video data/sample_videos/corridor.mp4 --algorithm ORB

# Expected:
# - Video plays smoothly
# - 3D trajectory displayed
# - No crashes
# - FPS ≥10 on 720p video (CPU)
```

### 3. Algorithm Verification (Manual)

- [ ] Test FAST: Keypoints detected, trajectory reasonable
- [ ] Test ORB: Keypoints detected, trajectory reasonable
- [ ] Test SIFT: Keypoints detected, trajectory reasonable
- [ ] Test Lucas-Kanade: Optical flow tracking smooth
- [ ] Switch algorithms mid-video: No crash, restart correctly

### 4. Cross-Platform Test

```bash
# macOS (M1/Intel)
python src/main.py --camera 0

# Windows (GPU)
python src/main.py --camera 0

# Expected: Both run smoothly, no platform-specific bugs
```

### 5. Performance Benchmarks

```bash
# Activate environment
conda activate vo

python scripts/benchmark.py --video data/sample_videos/test_720p.mp4

# Expected Output (CPU):
# FAST:  15-20 FPS
# ORB:   10-15 FPS
# SIFT:  6-10 FPS
# LK:    20-25 FPS (fastest)
```

### 6. CPU Performance Profiling

```bash
# Activate environment
conda activate vo

# Profile với cProfile
python -m cProfile -o profile.stats src/main.py --video data/sample_videos/test_720p.mp4 --algorithm ORB

# Analyze bottlenecks
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(20)"

# Expected: Feature detection và matching chiếm ≥60% runtime
```

### 7. Camera Calibration Test

```bash
# Activate environment
conda activate vo

python scripts/calibrate_camera.py --images data/calibration_images/ --output config/camera_params.yaml

# Expected:
# - Camera matrix K generated
# - Distortion coefficients saved
# - YAML file created
```

### 8. Code Quality

```bash
# Linting
pylint src/ --rcfile=.pylintrc

# Type checking
mypy src/ --strict

# Expected: No critical errors, score ≥8.0/10
```

### 9. Documentation Review

- [ ] README.md có Installation instructions rõ ràng
- [ ] README.md có Usage examples với screenshots
- [ ] ARCHITECTURE.md mô tả system design
- [ ] ALGORITHMS.md so sánh pros/cons từng algorithm
- [ ] CALIBRATION.md có step-by-step guide
- [ ] All public functions có docstrings

### 10. Production Readiness Checklist

- [ ] Error handling: Try/catch cho file I/O, camera access
- [ ] Logging: Debug info, errors logged properly
- [ ] Config: User có thể override parameters
- [ ] Graceful degradation: Fallback nếu GPU không có
- [ ] User feedback: Progress bar, status messages
- [ ] Exit handling: Proper cleanup (close camera, windows)

---

## ✅ DEFINITION OF DONE

Project được coi là **COMPLETE** khi:

1. ✅ Tất cả tasks P0-P10 đã complete
2. ✅ Phase X verification checklist 100% pass
3. ✅ Performance benchmarks đạt target (≥10 FPS cho ORB/720p trên CPU)
4. ✅ Cross-platform tests pass (macOS + Windows)
5. ✅ Code coverage ≥70%
6. ✅ Documentation đầy đủ (README, ARCHITECTURE, ALGORITHMS)
7. ✅ User có thể chạy app trong <5 phút từ clone repo

---

## 📌 Risk Mitigation

| Risk                       | Probability | Impact | Mitigation Strategy                                                  |
| -------------------------- | ----------- | ------ | -------------------------------------------------------------------- |
| CPU performance không đủ   | Medium      | High   | Optimize early (P8), reduce keypoints, downscale resolution nếu cần  |
| Monocular scale ambiguity  | High        | High   | Implement multiple scale estimation strategies, document limitations |
| Drift accumulation         | High        | High   | Add loop closure detection (future), benchmark drift metrics         |
| Cross-platform GUI issues  | Medium      | Medium | Use platform-agnostic framework (Tkinter safer than PyQt)            |
| SIFT patent issues         | Low         | Low    | SIFT free in OpenCV 4.4+, document license                           |
| Multi-threading complexity | Medium      | Medium | Thorough testing, use thread-safe data structures                    |

---

## 📝 Notes

- **Monocular VO limitations:** Scale ambiguity là fundamental problem. Trajectory shape đúng nhưng scale có thể sai. Document rõ limitation này.
- **CPU-only:** Đơn giản setup hơn GPU, nhưng FPS thấp hơn. Tối ưu bằng multi-threading và giảm số keypoints.
- **Algorithm tradeoffs:**
    - FAST: Nhanh nhất, ít descriptor info
    - ORB: Balance tốt speed/accuracy
    - SIFT: Chậm nhưng robust nhất
    - Lucas-Kanade: Nhanh, smooth tracking nhưng không handle large motion
- **Future enhancements:** Stereo VO, Loop closure, Bundle adjustment, Deep learning features (SuperPoint)

---

**Tạo bởi:** `project-planner` agent  
**Ngày:** 2026-01-30  
**Trạng thái:** 🟡 Waiting for user approval
