"""Trajectory Management

Quản lý và tích lũy trajectory của camera.
"""

import numpy as np
from typing import List, Tuple, Optional
from utils.logger import get_logger

logger = get_logger()


class Trajectory:
    """Quản lý 3D trajectory của camera"""

    def __init__(self, max_history: int = 0):
        """
        Initialize trajectory.

        Args:
            max_history: Maximum số poses lưu trữ (0 = unlimited)
        """
        self.max_history = max_history
        self.positions: List[np.ndarray] = []  # List of 3D positions (x, y, z)
        self.orientations: List[np.ndarray] = []  # List of rotation matrices

        # Current pose (cumulative transformation)
        self.current_position = np.zeros(3)  # [x, y, z]
        self.current_rotation = np.eye(3)  # 3x3 rotation matrix

        logger.info(f"Trajectory initialized: max_history={max_history}")

    def update(self, R: np.ndarray, t: np.ndarray) -> None:
        """
        Cập nhật trajectory với relative motion (R, t).

        Args:
            R: Rotation matrix 3x3 (từ frame trước → frame hiện tại)
            t: Translation vector 3x1 (normalized, scale ambiguous)
        """
        # Kiểm tra input validity
        if R.shape != (3, 3):
            logger.warning(f"Invalid rotation matrix shape: {R.shape}, skipping update")
            return

        if t.shape[0] != 3:
            logger.warning(f"Invalid translation vector shape: {t.shape}, skipping update")
            return

        # Flatten t nếu là (3, 1)
        t = t.flatten()

        # Update current pose: T_world = T_world * T_current
        # Rotation accumulation
        self.current_rotation = self.current_rotation @ R

        # Translation accumulation (rotate t to world frame first)
        self.current_position += self.current_rotation @ t

        # Lưu vào history
        self.positions.append(self.current_position.copy())
        self.orientations.append(self.current_rotation.copy())

        # Giới hạn history nếu cần
        if self.max_history > 0 and len(self.positions) > self.max_history:
            self.positions.pop(0)
            self.orientations.pop(0)

        logger.debug(f"Trajectory updated: position={self.current_position}, "
                     f"total_poses={len(self.positions)}")

    def get_positions_array(self) -> np.ndarray:
        """
        Lấy tất cả positions dưới dạng numpy array.

        Returns:
            Array shape (N, 3) với N là số poses
        """
        if len(self.positions) == 0:
            return np.zeros((0, 3))

        return np.array(self.positions)

    def get_current_position(self) -> np.ndarray:
        """Lấy current position"""
        return self.current_position.copy()

    def get_current_rotation(self) -> np.ndarray:
        """Lấy current rotation matrix"""
        return self.current_rotation.copy()

    def get_trajectory_length(self) -> int:
        """Lấy số poses trong trajectory"""
        return len(self.positions)

    def reset(self) -> None:
        """Reset trajectory về origin"""
        self.positions.clear()
        self.orientations.clear()
        self.current_position = np.zeros(3)
        self.current_rotation = np.eye(3)
        logger.info("Trajectory reset")

    def save_to_file(self, filename: str) -> None:
        """
        Lưu trajectory ra file text.

        Args:
            filename: Output file path

        Format: Mỗi dòng = x y z (space-separated)
        """
        positions = self.get_positions_array()

        if len(positions) == 0:
            logger.warning("Trajectory rỗng, không save")
            return

        np.savetxt(filename, positions, fmt='%.6f', delimiter=' ')
        logger.info(f"Đã lưu trajectory ({len(positions)} poses) vào {filename}")
