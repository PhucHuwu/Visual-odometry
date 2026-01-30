"""Camera Calibration Utilities

Load camera calibration parameters và undistort images.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from utils.logger import get_logger

logger = get_logger()


class CameraCalibration:
    """Quản lý camera calibration parameters"""

    def __init__(self, calibration_data: Dict[str, Any]):
        """
        Initialize từ calibration data dict.

        Args:
            calibration_data: Dict chứa camera_matrix, distortion_coefficients, image_size
        """
        self.calibration_data = calibration_data
        self._build_matrices()

    def _build_matrices(self) -> None:
        """Build camera matrix và distortion coefficients từ dict"""
        # Camera matrix K
        cam_data = self.calibration_data['camera_matrix']
        self.camera_matrix = np.array([
            [cam_data['fx'], 0, cam_data['cx']],
            [0, cam_data['fy'], cam_data['cy']],
            [0, 0, 1]
        ], dtype=np.float64)

        # Distortion coefficients
        dist_data = self.calibration_data['distortion_coefficients']
        self.dist_coeffs = np.array([
            dist_data['k1'],
            dist_data['k2'],
            dist_data['p1'],
            dist_data['p2'],
            dist_data['k3']
        ], dtype=np.float64)

        # Image size
        img_size = self.calibration_data['image_size']
        self.image_size = (img_size['width'], img_size['height'])

        logger.info(f"Camera calibration loaded: fx={cam_data['fx']:.2f}, "
                    f"fy={cam_data['fy']:.2f}, image_size={self.image_size}")

    def undistort(self, image: np.ndarray) -> np.ndarray:
        """
        Undistort image sử dụng calibration parameters.

        Args:
            image: Input image (distorted)

        Returns:
            Undistorted image
        """
        return cv2.undistort(image, self.camera_matrix, self.dist_coeffs)

    def get_camera_matrix(self) -> np.ndarray:
        """Lấy camera matrix K"""
        return self.camera_matrix.copy()

    def get_dist_coeffs(self) -> np.ndarray:
        """Lấy distortion coefficients"""
        return self.dist_coeffs.copy()

    @property
    def fx(self) -> float:
        """Focal length x"""
        return self.camera_matrix[0, 0]

    @property
    def fy(self) -> float:
        """Focal length y"""
        return self.camera_matrix[1, 1]

    @property
    def cx(self) -> float:
        """Principal point x"""
        return self.camera_matrix[0, 2]

    @property
    def cy(self) -> float:
        """Principal point y"""
        return self.camera_matrix[1, 2]


def calibrate_camera_from_images(
    images_path: str,
    pattern_size: Tuple[int, int] = (9, 6),
    square_size: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Calibrate camera từ checkerboard images.

    Args:
        images_path: Path tới folder chứa calibration images
        pattern_size: (width, height) của checkerboard inner corners
        square_size: Kích thước 1 ô vuông (mm hoặc arbitrary unit)

    Returns:
        (camera_matrix, dist_coeffs, image_size)

    Example:
        >>> K, dist, size = calibrate_camera_from_images("data/calibration_images")
    """
    logger.info(f"Bắt đầu camera calibration từ {images_path}")

    # Prepare object points (3D points in real world)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    obj_points = []  # 3D points
    img_points = []  # 2D points

    image_size = None

    # Tìm corners trong từng image
    image_files = list(Path(images_path).glob("*.jpg")) + list(Path(images_path).glob("*.png"))

    for img_file in image_files:
        img = cv2.imread(str(img_file))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        # Tìm chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            obj_points.append(objp)

            # Refine corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            img_points.append(corners_refined)

            logger.debug(f"Tìm thấy corners trong {img_file.name}")
        else:
            logger.warning(f"Không tìm thấy corners trong {img_file.name}")

    if len(obj_points) == 0:
        raise ValueError("Không tìm thấy checkerboard pattern nào!")

    logger.info(f"Tìm thấy {len(obj_points)} calibration images hợp lệ")

    # Calibrate camera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )

    if not ret:
        raise RuntimeError("Camera calibration thất bại!")

    logger.info("Camera calibration thành công!")
    logger.info(f"Camera matrix:\n{camera_matrix}")
    logger.info(f"Distortion coefficients: {dist_coeffs.ravel()}")

    return camera_matrix, dist_coeffs, image_size
