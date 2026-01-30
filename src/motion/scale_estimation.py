"""Scale Estimation

Xử lý monocular scale amb iguity.
"""

import numpy as np
from typing import Optional
from utils.logger import get_logger

logger = get_logger()


class ScaleEstimator:
    """Estimate scale cho monocular VO"""

    def __init__(self, method: str = "constant"):
        """
        Initialize scale estimator.

        Args:
            method: Scale estimation method
                - "constant": Giả định constant velocity
                - "unit": Luôn dùng unit scale (t không scale)
                - "median": Median depth estimation (experimental)
        """
        self.method = method
        self.previous_scale = 1.0

        logger.info(f"Scale estimator initialized: method={method}")

    def estimate(
        self,
        t: np.ndarray,
        previous_t: Optional[np.ndarray] = None
    ) -> float:
        """
        Estimate scale cho translation vector.

        Args:
            t: Current translation vector (unit)
            previous_t: Previous translation vector (optional)

        Returns:
            Estimated scale factor

        Note:
            Monocular VO có scale ambiguity - không thể xác định
            absolute scale. Methods này đều là heuristics.
        """
        if self.method == "constant":
            # Giả định constant velocity
            scale = self.previous_scale

        elif self.method == "unit":
            # Luôn dùng unit scale
            scale = 1.0

        elif self.method == "median":
            # Heuristic: giả định median translation magnitude
            if previous_t is not None:
                prev_norm = np.linalg.norm(previous_t)
                curr_norm = np.linalg.norm(t)

                if prev_norm > 1e-6 and curr_norm > 1e-6:
                    scale = prev_norm / curr_norm
                else:
                    scale = self.previous_scale
            else:
                scale = 1.0

        else:
            logger.warning(f"Unknown scale method: {self.method}, using unit scale")
            scale = 1.0

        # Cập nhật previous scale
        self.previous_scale = scale

        return scale

    def apply_scale(self, t: np.ndarray, scale: float) -> np.ndarray:
        """
        Áp dụng scale lên translation vector.

        Args:
            t: Translation vector (unit)
            scale: Scale factor

        Returns:
            Scaled translation vector
        """
        return t * scale
