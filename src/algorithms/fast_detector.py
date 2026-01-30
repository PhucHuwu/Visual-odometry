"""FAST Feature Detector

Features from Accelerated Segment Test (FAST) corner detection.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
from .base_algorithm import BaseAlgorithm
from ..utils.logger import get_logger

logger = get_logger()


class FASTDetector(BaseAlgorithm):
    """FAST corner detector với BFMatcher"""

    def __init__(self, config: dict):
        """
        Initialize FAST detector.

        Args:
            config: Dict với keys: threshold, nonmaxSuppression, type
        """
        super().__init__(config)

        self.threshold = config.get('threshold', 20)
        self.nonmax_suppression = config.get('nonmaxSuppression', True)
        self.detector_type = config.get('type', cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)

        # Tạo FAST detector
        self.detector = cv2.FastFeatureDetector_create(
            threshold=self.threshold,
            nonmaxSuppression=self.nonmax_suppression,
            type=self.detector_type
        )

        # Tạo ORB descriptor extractor (FAST không có descriptor riêng)
        self.descriptor_extractor = cv2.ORB_create()

        # Tạo BFMatcher
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        logger.info(f"FAST detector initialized: threshold={self.threshold}")

    def detect(self, image: np.ndarray) -> Tuple[List, Optional[np.ndarray]]:
        """
        Detect FAST corners và compute ORB descriptors.

        Args:
            image: Grayscale image

        Returns:
            (keypoints, descriptors)
        """
        # Detect keypoints
        keypoints = self.detector.detect(image, None)

        # Compute ORB descriptors cho keypoints
        keypoints, descriptors = self.descriptor_extractor.compute(image, keypoints)

        logger.debug(f"FAST detected {len(keypoints)} keypoints")

        return keypoints, descriptors

    def match(
        self,
        keypoints1: List,
        descriptors1: np.ndarray,
        keypoints2: List,
        descriptors2: np.ndarray
    ) -> List:
        """
        Match descriptors sử dụng BFMatcher với ratio test.

        Args:
            keypoints1, descriptors1: Frame 1
            keypoints2, descriptors2: Frame 2

        Returns:
            List of good matches
        """
        if descriptors1 is None or descriptors2 is None:
            return []

        # KNN matching
        matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)

        # Lowe's ratio test
        good_matches = []
        ratio_threshold = self.config.get('ratio_test', 0.7)

        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)

        logger.debug(f"FAST matched {len(good_matches)}/{len(matches)} keypoints")

        return good_matches
