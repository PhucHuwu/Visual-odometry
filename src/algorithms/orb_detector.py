"""ORB Feature Detector

Oriented FAST and Rotated BRIEF (ORB) feature detection and description.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
from .base_algorithm import BaseAlgorithm
from ..utils.logger import get_logger

logger = get_logger()


class ORBDetector(BaseAlgorithm):
    """ORB feature detector và descriptor"""

    def __init__(self, config: dict):
        """
        Initialize ORB detector.

        Args:
            config: Dict với ORB parameters
        """
        super().__init__(config)

        # Tạo ORB detector
        self.detector = cv2.ORB_create(
            nfeatures=config.get('nfeatures', 500),
            scaleFactor=config.get('scaleFactor', 1.2),
            nlevels=config.get('nlevels', 8),
            edgeThreshold=config.get('edgeThreshold', 31),
            firstLevel=config.get('firstLevel', 0),
            WTA_K=config.get('WTA_K', 2),
            patchSize=config.get('patchSize', 31)
        )

        # BFMatcher với Hamming distance
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        logger.info(f"ORB detector initialized: nfeatures={config.get('nfeatures', 500)}")

    def detect(self, image: np.ndarray) -> Tuple[List, Optional[np.ndarray]]:
        """
        Detect và describe ORB features.

        Args:
            image: Grayscale image

        Returns:
            (keypoints, descriptors)
        """
        keypoints, descriptors = self.detector.detectAndCompute(image, None)

        logger.debug(f"ORB detected {len(keypoints)} keypoints")

        return keypoints, descriptors

    def match(
        self,
        keypoints1: List,
        descriptors1: np.ndarray,
        keypoints2: List,
        descriptors2: np.ndarray
    ) -> List:
        """
        Match ORB descriptors với ratio test.

        Args:
            keypoints1, descriptors1: Frame 1
            keypoints2, descriptors2: Frame 2

        Returns:
            List of good matches
        """
        if descriptors1 is None or descriptors2 is None:
            return []

        # KNN matching (k=2 for ratio test)
        matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)

        # Lowe's ratio test
        good_matches = []
        ratio_threshold = self.config.get('ratio_test', 0.7)

        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)

        logger.debug(f"ORB matched {len(good_matches)}/{len(matches)} keypoints")

        return good_matches
