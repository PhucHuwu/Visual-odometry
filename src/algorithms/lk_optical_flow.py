"""Lucas-Kanade Optical Flow

Sparse optical flow tracking sử dụng Lucas-Kanade method.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
from algorithms.base_algorithm import BaseAlgorithm
from utils.logger import get_logger

logger = get_logger()


class LucasKanadeOpticalFlow(BaseAlgorithm):
    """Lucas-Kanade optical flow tracking"""

    def __init__(self, config: dict):
        """
        Initialize Lucas-Kanade tracker.

        Args:
            config: Dict với LK parameters
        """
        super().__init__(config)

        # Parameters cho Lucas-Kanade
        self.win_size = tuple(config.get('winSize', [21, 21]))
        self.max_level = config.get('maxLevel', 3)

        criteria_config = config.get('criteria', {})
        self.criteria = (
            criteria_config.get('type', cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT),
            criteria_config.get('maxCount', 30),
            criteria_config.get('epsilon', 0.01)
        )

        # Tạo FAST detector cho initial keypoints
        self.fast_detector = cv2.FastFeatureDetector_create(threshold=20)

        # Lưu previous frame và keypoints
        self.prev_image: Optional[np.ndarray] = None
        self.prev_keypoints: Optional[List] = None

        logger.info(f"Lucas-Kanade initialized: winSize={self.win_size}, "
                    f"maxLevel={self.max_level}")

    def detect(self, image: np.ndarray) -> Tuple[List, Optional[np.ndarray]]:
        """
        Detect keypoints sử dụng FAST (cho initial frame).

        Args:
            image: Grayscale image

        Returns:
            (keypoints, None) - LK không có descriptors
        """
        keypoints = self.fast_detector.detect(image, None)

        logger.debug(f"Lucas-Kanade detected {len(keypoints)} initial keypoints")

        return keypoints, None

    def track(
        self,
        prev_image: np.ndarray,
        curr_image: np.ndarray,
        prev_points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Track keypoints từ prev_image sang curr_image.

        Args:
            prev_image: Previous grayscale image
            curr_image: Current grayscale image
            prev_points: Points to track, shape (N, 1, 2)

        Returns:
            (curr_points, status, error) tuple
            curr_points: Tracked points in current image
            status: 1 nếu tracked successfully, 0 nếu lost
            error: Tracking error
        """
        curr_points, status, error = cv2.calcOpticalFlowPyrLK(
            prev_image,
            curr_image,
            prev_points,
            None,
            winSize=self.win_size,
            maxLevel=self.max_level,
            criteria=self.criteria
        )

        return curr_points, status, error

    def match(
        self,
        keypoints1: List,
        descriptors1: np.ndarray,
        keypoints2: List,
        descriptors2: np.ndarray
    ) -> List:
        """
        LK không dùng matching - override để compatibility.

        Returns:
            Empty list (matching không áp dụng cho optical flow)
        """
        logger.warning("match() không áp dụng cho Lucas-Kanade optical flow")
        return []

    def track_and_filter(
        self,
        prev_image: np.ndarray,
        curr_image: np.ndarray,
        prev_keypoints: List
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Track keypoints và filter outliers.

        Args:
            prev_image: Previous grayscale image
            curr_image: Current grayscale image
            prev_keypoints: List of cv2.KeyPoint

        Returns:
            (prev_points, curr_points) tuple - chỉ good tracks
        """
        # Convert keypoints to points
        prev_pts = np.array([kp.pt for kp in prev_keypoints], dtype=np.float32)
        prev_pts = prev_pts.reshape(-1, 1, 2)

        # Track
        curr_pts, status, error = self.track(prev_image, curr_image, prev_pts)

        # Filter good tracks
        good_prev = prev_pts[status.flatten() == 1]
        good_curr = curr_pts[status.flatten() == 1]

        logger.debug(f"Lucas-Kanade tracked {len(good_curr)}/{len(prev_keypoints)} points")

        return good_prev.reshape(-1, 2), good_curr.reshape(-1, 2)
