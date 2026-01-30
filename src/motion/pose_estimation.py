"""Pose Estimation

Recover camera pose (R, t) từ Essential Matrix.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from utils.logger import get_logger

logger = get_logger()


def recover_pose(
    E: np.ndarray,
    points1: np.ndarray,
    points2: np.ndarray,
    camera_matrix: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    """
    Recover camera pose (R, t) từ Essential Matrix.

    Args:
        E: Essential matrix 3x3
        points1: Points từ frame 1, shape (N, 2)
        points2: Points từ frame 2, shape (N, 2)
        camera_matrix: Camera intrinsic matrix K
        mask: Inlier mask từ Essential Matrix estimation

    Returns:
        (R, t, num_inliers) tuple
        R: Rotation matrix 3x3, hoặc None nếu thất bại
        t: Translation vector 3x1 (unit vector), hoặc None nếu thất bại
        num_inliers: Số inliers sau khi recover pose

    Note:
        Translation t là unit vector (|t| = 1) do monocular scale ambiguity
    """
    # Recover pose
    num_inliers, R, t, pose_mask = cv2.recoverPose(
        E,
        points1,
        points2,
        camera_matrix,
        mask=mask
    )

    if R is None or t is None:
        logger.warning("Không recover được pose")
        return None, None, 0

    # Validate rotation matrix
    if not is_valid_rotation_matrix(R):
        logger.warning("Rotation matrix không hợp lệ")
        return None, None, 0

    # Normalize translation (đảm bảo unit vector)
    t_norm = np.linalg.norm(t)
    if t_norm > 1e-6:
        t = t / t_norm

    logger.debug(f"Pose recovered: {num_inliers} points, "
                 f"R_det={np.linalg.det(R):.4f}, t_norm={np.linalg.norm(t):.4f}")

    return R, t, num_inliers


def is_valid_rotation_matrix(R: np.ndarray, tolerance: float = 1e-3) -> bool:
    """
    Kiểm tra R có phải là valid rotation matrix không.

    Rotation matrix phải thỏa mãn:
    1. det(R) = 1 (not -1, which is reflection)
    2. R^T * R = I (orthogonal)

    Args:
        R: Matrix 3x3
        tolerance: Tolerance cho numerical errors

    Returns:
        True nếu R là valid rotation matrix
    """
    # Check determinant = 1
    det = np.linalg.det(R)
    if abs(det - 1.0) > tolerance:
        logger.debug(f"det(R) = {det:.6f} (expected 1.0)")
        return False

    # Check orthogonality: R^T * R = I
    should_be_identity = R.T @ R
    identity = np.eye(3)

    if not np.allclose(should_be_identity, identity, atol=tolerance):
        logger.debug(f"R^T * R không phải identity matrix")
        return False

    return True


def decompose_essential_matrix(E: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose Essential Matrix thành 4 possible (R, t) solutions.

    Args:
        E: Essential matrix 3x3

    Returns:
        (R_solutions, t_solutions) tuple
        R_solutions: List of 2 rotation matrices
        t_solutions: List of 2 translation vectors

    Note:
        Essential Matrix có 4 possible solutions:
        (R1, t), (R1, -t), (R2, t), (R2, -t)
        Chỉ có 1 solution đúng (points ở phía trước cả 2 cameras)
    """
    # SVD decomposition
    U, _, Vt = np.linalg.svd(E)

    # Đảm bảo det(U) = det(V) = 1
    if np.linalg.det(U) < 0:
        U *= -1
    if np.linalg.det(Vt) < 0:
        Vt *= -1

    # W matrix
    W = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])

    # 2 possible rotations
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt

    # 1 possible translation (up to sign)
    t = U[:, 2].reshape(3, 1)

    return [R1, R2], [t, -t]
