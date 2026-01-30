"""Camera Input Handler

Xử lý input từ webcam hoặc video file.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from ..utils.logger import get_logger

logger = get_logger()


class CameraInput:
    """Handle camera input từ webcam hoặc video file"""

    def __init__(
        self,
        source: int | str = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None
    ):
        """
        Initialize camera input.

        Args:
            source: Camera device ID (int) hoặc video file path (str)
            width: Resize width (None = giữ nguyên)
            height: Resize height (None = giữ nguyên)
            fps: Target FPS (None = camera default)
        """
        self.source = source
        self.target_width = width
        self.target_height = height
        self.target_fps = fps

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_live = isinstance(source, int)
        self.frame_count = 0
        self.total_frames = 0

        self._open()

    def _open(self) -> None:
        """Mở video capture"""
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            raise RuntimeError(f"Không thể mở video source: {self.source}")

        # Lấy thông tin video
        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)

        if not self.is_live:
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Set FPS nếu được chỉ định (chỉ hoạt động với một số cameras)
        if self.target_fps is not None and self.is_live:
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        source_type = "webcam" if self.is_live else "video file"
        logger.info(f"Đã mở {source_type}: {self.source}")
        logger.info(f"Resolution: {self.original_width}x{self.original_height}, "
                    f"FPS: {self.original_fps:.1f}")

        if not self.is_live:
            logger.info(f"Tổng số frames: {self.total_frames}")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Đọc frame từ camera.

        Returns:
            (success, frame) tuple
            success: True nếu đọc thành công
            frame: BGR image hoặc None nếu thất bại
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()

        if not ret:
            return False, None

        self.frame_count += 1

        # Resize nếu cần
        if self.target_width is not None and self.target_height is not None:
            frame = cv2.resize(frame, (self.target_width, self.target_height))

        return True, frame

    def get_fps(self) -> float:
        """Lấy FPS của video"""
        return self.original_fps if self.cap else 0.0

    def get_frame_count(self) -> int:
        """Lấy số frame đã đọc"""
        return self.frame_count

    def get_progress(self) -> float:
        """
        Lấy progress (0.0 - 1.0) cho video file.

        Returns:
            Progress (0.0 - 1.0), hoặc 0.0 nếu là live camera
        """
        if self.is_live or self.total_frames == 0:
            return 0.0

        return min(self.frame_count / self.total_frames, 1.0)

    def release(self) -> None:
        """Giải phóng video capture"""
        if self.cap is not None:
            self.cap.release()
            logger.info(f"Đã release camera input. Tổng frames đã đọc: {self.frame_count}")
            self.cap = None

    def __enter__(self):
        """Context manager support"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        self.release()

    def __del__(self):
        """Destructor"""
        self.release()
