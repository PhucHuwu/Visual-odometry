"""Visual Odometry Pipeline

Main orchestrator cho VO system.
"""

import cv2
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path

from core.camera import CameraInput
from core.preprocessor import Preprocessor
from core.trajectory import Trajectory
from algorithms.fast_detector import FASTDetector
from algorithms.orb_detector import ORBDetector
from algorithms.sift_detector import SIFTDetector
from algorithms.lk_optical_flow import LucasKanadeOpticalFlow
from motion.essential_matrix import estimate_essential_matrix
from motion.pose_estimation import recover_pose
from motion.scale_estimation import ScaleEstimator
from utils.config_loader import ConfigLoader
from utils.calibration import CameraCalibration
from utils.logger import get_logger

logger = get_logger()


class VOPipeline:
    """Main Visual Odometry Pipeline"""

    def __init__(self, config: ConfigLoader):
        """
        Initialize VO Pipeline.

        Args:
            config: ConfigLoader instance
        """
        self.config = config

        # Load camera calibration
        calib_file = config.get('camera.calibration_file')
        calib_data = config.load_additional(calib_file)
        self.calibration = CameraCalibration(calib_data)

        # Load algorithm config
        algo_config_file = config.get('algorithm.config_file')
        algo_configs = config.load_additional(algo_config_file)

        # Lấy matching config
        self.matching_config = algo_configs.get('matching', {})

        # Initialize algorithm
        algo_type = config.get('algorithm.type', 'orb')
        self.algorithm = self._create_algorithm(algo_type, algo_configs)

        # Initialize components
        self.preprocessor = Preprocessor(apply_denoise=False)
        self.trajectory = Trajectory(max_history=config.get('visualization.trajectory_history', 1000))
        self.scale_estimator = ScaleEstimator(method="unit")

        # State
        self.prev_image: Optional[np.ndarray] = None
        self.prev_keypoints: Optional[list] = None
        self.prev_descriptors: Optional[np.ndarray] = None

        self.frame_count = 0
        self.total_keypoints = 0
        self.total_matches = 0

        logger.info(f"VO Pipeline initialized với algorithm: {algo_type}")

    def _create_algorithm(self, algo_type: str, algo_configs: Dict[str, Any]):
        """Tạo algorithm instance dựa trên type"""
        algo_type = algo_type.lower()

        # Merge matching config vào algo config
        if algo_type == 'fast':
            config = {**algo_configs['fast'], **self.matching_config}
            return FASTDetector(config)
        elif algo_type == 'orb':
            config = {**algo_configs['orb'], **self.matching_config}
            return ORBDetector(config)
        elif algo_type == 'sift':
            config = {**algo_configs['sift'], **self.matching_config}
            return SIFTDetector(config)
        elif algo_type == 'lucas_kanade':
            config = {**algo_configs['lucas_kanade'], **self.matching_config}
            return LucasKanadeOpticalFlow(config)
        else:
            logger.warning(f"Unknown algorithm: {algo_type}, using ORB")
            config = {**algo_configs['orb'], **self.matching_config}
            return ORBDetector(config)

    def process_frame(self, frame: np.ndarray) -> bool:
        """
        Xử lý một frame.

        Args:
            frame: Input BGR frame

        Returns:
            True nếu xử lý thành công, False nếu không
        """
        self.frame_count += 1

        # Undistort nếu cần
        if self.config.get('camera.use_calibration', True):
            frame = self.calibration.undistort(frame)

        # Preprocessing
        gray = self.preprocessor.process(frame)

        # First frame: chỉ detect keypoints
        if self.prev_image is None:
            self.prev_image = gray
            self.prev_keypoints, self.prev_descriptors = self.algorithm.detect(gray)
            logger.debug(f"Frame {self.frame_count}: Detected {len(self.prev_keypoints)} initial keypoints")
            return False  # Chưa estimate được motion

        # Detect/track keypoints
        if isinstance(self.algorithm, LucasKanadeOpticalFlow):
            # Optical flow tracking
            pts1, pts2 = self.algorithm.track_and_filter(
                self.prev_image,
                gray,
                self.prev_keypoints
            )
        else:
            # Feature matching
            curr_keypoints, curr_descriptors = self.algorithm.detect(gray)

            if curr_descriptors is None or self.prev_descriptors is None:
                logger.warning(f"Frame {self.frame_count}: No descriptors")
                return False

            matches = self.algorithm.match(
                self.prev_keypoints,
                self.prev_descriptors,
                curr_keypoints,
                curr_descriptors
            )

            if len(matches) < self.matching_config.get('min_matches', 10):
                logger.warning(f"Frame {self.frame_count}: Không đủ matches ({len(matches)})")
                # Update previous frame
                self.prev_image = gray
                self.prev_keypoints = curr_keypoints
                self.prev_descriptors = curr_descriptors
                return False

            pts1, pts2 = self.algorithm.extract_matched_points(
                self.prev_keypoints,
                curr_keypoints,
                matches
            )

            self.total_matches += len(matches)

        self.total_keypoints += len(pts1)

        # Estimate Essential Matrix
        E, mask = estimate_essential_matrix(
            pts1,
            pts2,
            self.calibration.get_camera_matrix(),
            ransac_threshold=self.matching_config.get('ransac_threshold', 1.0),
            ransac_confidence=self.matching_config.get('ransac_confidence', 0.999)
        )

        if E is None:
            logger.warning(f"Frame {self.frame_count}: Không estimate được Essential Matrix")
            return False

        # Recover pose
        R, t, num_inliers = recover_pose(
            E,
            pts1,
            pts2,
            self.calibration.get_camera_matrix(),
            mask
        )

        if R is None or t is None:
            logger.warning(f"Frame {self.frame_count}: Không recover được pose")
            return False

        # Scale estimation (monocular)
        scale = self.scale_estimator.estimate(t)
        t_scaled = self.scale_estimator.apply_scale(t, scale)

        # Update trajectory
        self.trajectory.update(R, t_scaled)

        logger.debug(f"Frame {self.frame_count}: Processed successfully, "
                     f"{num_inliers} inliers, trajectory_length={self.trajectory.get_trajectory_length()}")

        # Update previous state
        self.prev_image = gray

        if not isinstance(self.algorithm, LucasKanadeOpticalFlow):
            self.prev_keypoints = curr_keypoints
            self.prev_descriptors = curr_descriptors
        else:
            # Re-detect keypoints cho LK nếu số keypoints giảm quá nhiều
            if len(pts2) < 50:
                self.prev_keypoints, _ = self.algorithm.detect(gray)

        return True

    def get_trajectory(self) -> np.ndarray:
        """Lấy trajectory array"""
        return self.trajectory.get_positions_array()

    def get_stats(self) -> Dict[str, Any]:
        """Lấy statistics"""
        return {
            'frame_count': self.frame_count,
            'trajectory_length': self.trajectory.get_trajectory_length(),
            'total_keypoints': self.total_keypoints,
            'total_matches': self.total_matches,
            'avg_keypoints_per_frame': self.total_keypoints / max(self.frame_count, 1),
            'current_position': self.trajectory.get_current_position()
        }

    def reset(self) -> None:
        """Reset pipeline state"""
        self.prev_image = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.frame_count = 0
        self.total_keypoints = 0
        self.total_matches = 0
        self.trajectory.reset()
        logger.info("VO Pipeline reset")

    def save_trajectory(self, filename: str) -> None:
        """Lưu trajectory ra file"""
        self.trajectory.save_to_file(filename)
