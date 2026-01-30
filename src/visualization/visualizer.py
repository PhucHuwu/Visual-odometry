"""Visualization Module

Hiển thị camera feed và keypoints.
"""

import cv2
import numpy as np
from typing import Optional, List


class Visualizer:
    """Visualize camera feed và VO results"""

    def __init__(self, window_name: str = "Visual Odometry"):
        """
        Initialize visualizer.

        Args:
            window_name: Tên cửa sổ hiển thị
        """
        self.window_name = window_name
        self.show_display = True

    def draw_keypoints(
        self,
        frame: np.ndarray,
        keypoints: List,
        color: tuple = (0, 255, 0)
    ) -> np.ndarray:
        """
        Vẽ keypoints lên frame.

        Args:
            frame: BGR image
            keypoints: List of cv2.KeyPoint
            color: BGR color tuple (default: green)

        Returns:
            Frame với keypoints được vẽ
        """
        frame_copy = frame.copy()

        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            cv2.circle(frame_copy, (x, y), 3, color, -1)

        return frame_copy

    def draw_matches(
        self,
        frame: np.ndarray,
        points1: np.ndarray,
        points2: np.ndarray
    ) -> np.ndarray:
        """
        Vẽ matched points với optical flow arrows.

        Args:
            frame: Current BGR frame
            points1: Previous points (N, 2)
            points2: Current points (N, 2)

        Returns:
            Frame với flow arrows
        """
        frame_copy = frame.copy()

        for pt1, pt2 in zip(points1, points2):
            x1, y1 = int(pt1[0]), int(pt1[1])
            x2, y2 = int(pt2[0]), int(pt2[1])

            # Draw arrow từ pt1 đến pt2
            cv2.arrowedLine(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 1, tipLength=0.3)
            cv2.circle(frame_copy, (x2, y2), 2, (0, 0, 255), -1)

        return frame_copy

    def add_stats(
        self,
        frame: np.ndarray,
        stats: dict
    ) -> np.ndarray:
        """
        Thêm statistics text lên frame.

        Args:
            frame: BGR image
            stats: Dict với keys: fps, frame_count, keypoints, trajectory_length, etc.

        Returns:
            Frame với stats text
        """
        frame_copy = frame.copy()

        # Background overlay cho text
        overlay = frame_copy.copy()
        cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
        frame_copy = cv2.addWeighted(overlay, 0.6, frame_copy, 0.4, 0)

        # Draw stats
        y = 35
        font = cv2.FONT_HERSHEY_SIMPLEX

        texts = [
            f"Frame: {stats.get('frame_count', 0)}",
            f"FPS: {stats.get('fps', 0):.1f}",
            f"Keypoints: {stats.get('keypoints', 0)}",
            f"Trajectory: {stats.get('trajectory_length', 0)} poses",
            f"Algorithm: {stats.get('algorithm', 'N/A')}"
        ]

        for text in texts:
            cv2.putText(frame_copy, text, (20, y), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            y += 25

        return frame_copy

    def show(self, frame: np.ndarray, wait_ms: int = 1) -> bool:
        """
        Hiển thị frame.

        Args:
            frame: BGR image to display
            wait_ms: Wait time in ms (default: 1ms)

        Returns:
            False nếu user nhấn 'q' để quit
        """
        if not self.show_display:
            return True

        cv2.imshow(self.window_name, frame)

        key = cv2.waitKey(wait_ms) & 0xFF
        if key == ord('q'):
            return False

        return True

    def close(self) -> None:
        """Đóng cửa sổ"""
        cv2.destroyAllWindows()
