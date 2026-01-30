"""Image Preprocessing

Chuyển đổi và tiền xử lý ảnh cho VO pipeline.
"""

import cv2
import numpy as np
from typing import Optional
from utils.logger import get_logger

logger = get_logger()


class Preprocessor:
    """Tiền xử lý ảnh: grayscale conversion, denoising"""

    def __init__(self, apply_denoise: bool = False, denoise_strength: int = 3):
        """
        Initialize preprocessor.

        Args:
            apply_denoise: Có áp dụng Gaussian blur để denoise không
            denoise_strength: Kernel size cho Gaussian blur (phải là số lẻ)
        """
        self.apply_denoise = apply_denoise

        # Ensure kernel size là số lẻ
        if denoise_strength % 2 == 0:
            denoise_strength += 1

        self.denoise_kernel = (denoise_strength, denoise_strength)

        logger.debug(f"Preprocessor initialized: denoise={apply_denoise}, "
                     f"kernel={self.denoise_kernel}")

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Xử lý ảnh: BGR -> Grayscale -> (optional) Denoise.

        Args:
            image: Input BGR image

        Returns:
            Grayscale image (uint8)
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image  # Đã là grayscale

        # Apply denoise nếu cần
        if self.apply_denoise:
            gray = cv2.GaussianBlur(gray, self.denoise_kernel, 0)

        return gray

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Callable shortcut cho process()"""
        return self.process(image)
