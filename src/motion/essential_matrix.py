"""Essential Matrix Estimation

Tính toán Essential Matrix từ matched points.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from ..utils.logger import get_logger

logger = get_logger()


def estimate_essential_matrix(
    points1: np.ndarray,
    points2: np.ndarray,
    camera_matrix: np.ndarray,
    ransac_threshold: float = 1.0,
    ransac_confidence: float = 0.999
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Estimate Essential Matrix từ matched points.

    Args:
        points1: Points từ frame 1, shape (N, 2)
        points2: Points từ frame 2, shape (N, 2)
        camera_matrix: Camera intrinsic matrix K (3x3)
        ransac_threshold: RANSAC threshold (pixels)
        ransac_confidence: RANSAC confidence (0.0-1.0)

    Returns:
        (E, mask) tuple
        E: Essential matrix 3x3, hoặc None nếu thất bại
        mask: Inlier mask, hoặc None nếu thất bại

    Note:
        Essential Matrix E thỏa mãn: p2^T * E * p1 = 0
        với p1, p2 là normalized image coordinates
    """
    if len(points1) < 5 or len(points2) < 5:
        logger.warning(f"Không đủ points để estimate E matrix: {len(points1)} points")
        return None, None

    # Estimate Essential Matrix với RANSAC
    E, mask = cv2.findEssentialMat(
        points1,
        points2,
        camera_matrix,
        method=cv2.RANSAC,
        prob=ransac_confidence,
        threshold=ransac_threshold
    )

    if E is None:
        logger.warning("Không estimate được Essential Matrix")
        return None, None

    # Đếm inliers
    num_inliers = np.sum(mask) if mask is not None else 0
    inlier_ratio = num_inliers / len(points1) if len(points1) > 0 else 0.0

    logger.debug(f"Essential Matrix estimated: {num_inliers}/{len(points1)} inliers "
                 f"({inlier_ratio*100:.1f}%)")

    return E, mask


def check_essential_matrix(E: np.ndarray, tolerance: float = 1e-5) -> bool:
    """
    Kiểm tra tính hợp lệ của Essential Matrix.

    Essential Matrix phải thỏa mãn:
    1. Rank = 2
    2. 2*E*E^T*E - trace(E*E^T)*E = 0 (Demazure constraint)

    Args:
        E: Essential matrix 3x3
        tolerance: Tolerance cho numerical errors

    Returns:
        True nếu E hợp lệ
    """
    # Check rank = 2
    _, s, _ = np.linalg.svd(E)

    # Singular values phải có dạng [σ, σ, 0] (với tolerance)
    if abs(s[0] - s[1]) > tolerance or abs(s[2]) > tolerance:
        logger.warning(f"Essential Matrix có singular values không hợp lệ: {s}")
        return False

    logger.debug("Essential Matrix validation passed")
    return True
