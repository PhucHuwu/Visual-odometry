"""Base Algorithm Interface

Abstract base class cho tất cả feature detection algorithms.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, List, Optional


class BaseAlgorithm(ABC):
    """Abstract base class cho feature detection/tracking algorithms"""

    def __init__(self, config: dict):
        """
        Initialize algorithm với config.

        Args:
            config: Dictionary chứa algorithm parameters
        """
        self.config = config
        self.name = self.__class__.__name__

    @abstractmethod
    def detect(self, image: np.ndarray) -> Tuple[List, Optional[np.ndarray]]:
        """
        Phát hiện keypoints trong ảnh.

        Args:
            image: Grayscale image

        Returns:
            (keypoints, descriptors) tuple
            keypoints: List of cv2.KeyPoint
            descriptors: numpy array hoặc None nếu không có
        """
        pass

    @abstractmethod
    def match(
        self,
        keypoints1: List,
        descriptors1: np.ndarray,
        keypoints2: List,
        descriptors2: np.ndarray
    ) -> List:
        """
        Match keypoints giữa 2 frames.

        Args:
            keypoints1: Keypoints từ frame 1
            descriptors1: Descriptors từ frame 1
            keypoints2: Keypoints từ frame 2
            descriptors2: Descriptors từ frame 2

        Returns:
            List of cv2.DMatch objects
        """
        pass

    def extract_matched_points(
        self,
        keypoints1: List,
        keypoints2: List,
        matches: List
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Trích xuất matched points từ matches.

        Args:
            keypoints1: Keypoints từ frame 1
            keypoints2: Keypoints từ frame 2
            matches: List of DMatch

        Returns:
            (points1, points2) tuple
            points1: numpy array shape (N, 2)
            points2: numpy array shape (N, 2)
        """
        pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

        return pts1, pts2

    def __repr__(self) -> str:
        return f"{self.name}()"
