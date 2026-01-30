"""SIFT Feature Detector

Scale-Invariant Feature Transform (SIFT) feature detection and description.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
from .base_algorithm import BaseAlgorithm
from ..utils.logger import get_logger

logger = get_logger()


class SIFTDetector(BaseAlgorithm):
    """SIFT feature detector và descriptor"""

    def __init__(self, config: dict):
        """
        Initialize SIFT detector.

        Args:
            config: Dict với SIFT parameters
        """
        super().__init__(config)

        # Tạo SIFT detector
        self.detector = cv2.SIFT_create(
            nfeatures=config.get('nfeatures', 0),
            nOctaveLayers=config.get('nOctaveLayers', 3),
            contrastThreshold=config.get('contrastThreshold', 0.04),
            edgeThreshold=config.get('edgeThreshold', 10),
            sigma=config.get('sigma', 1.6)
        )

        # BFMatcher với L2 distance (SIFT descriptors là float)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        logger.info(f"SIFT detector initialized: nfeatures={config.get('nfeatures', 0)}")

    def detect(self, image: np.ndarray) -> Tuple[List, Optional[np.ndarray]]:
        """
        Detect và describe SIFT features.

        Args:
            image: Grayscale image

        Returns:
            (keypoints, descriptors)
        """
        keypoints, descriptors = self.detector.detectAndCompute(image, None)

        logger.debug(f"SIFT detected {len(keypoints)} keypoints")

        return keypoints, descriptors

    def match(
        self,
        keypoints1: List,
        descriptors1: np.ndarray,
        keypoints2: List,
        descriptors2: np.ndarray
    ) -> List:
        """
        Match SIFT descriptors với ratio test.

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

        logger.debug(f"SIFT matched {len(good_matches)}/{len(matches)} keypoints")

        return good_matches
