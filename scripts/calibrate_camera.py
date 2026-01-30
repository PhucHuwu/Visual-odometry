"""Camera Calibration Script

Calibrate camera từ checkerboard images.
"""

from utils.logger import setup_logger
from utils.calibration import calibrate_camera_from_images
import yaml
import argparse
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).resolve().parent.parent / 'src'
sys.path.insert(0, str(src_path))


def save_calibration_to_yaml(
    camera_matrix,
    dist_coeffs,
    image_size,
    output_file
):
    """Lưu calibration parameters vào YAML file"""
    data = {
        'camera_matrix': {
            'fx': float(camera_matrix[0, 0]),
            'fy': float(camera_matrix[1, 1]),
            'cx': float(camera_matrix[0, 2]),
            'cy': float(camera_matrix[1, 2])
        },
        'distortion_coefficients': {
            'k1': float(dist_coeffs[0]),
            'k2': float(dist_coeffs[1]),
            'p1': float(dist_coeffs[2]),
            'p2': float(dist_coeffs[3]),
            'k3': float(dist_coeffs[4])
        },
        'image_size': {
            'width': int(image_size[0]),
            'height': int(image_size[1])
        }
    }

    # Lưu file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Camera Calibration from Checkerboard Images'
    )

    parser.add_argument(
        '--images',
        type=str,
        required=True,
        help='Path tới folder chứa calibration images'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='config/camera_params.yaml',
        help='Output YAML file (default: config/camera_params.yaml)'
    )

    parser.add_argument(
        '--pattern-width',
        type=int,
        default=9,
        help='Số inner corners theo chiều ngang (default: 9)'
    )

    parser.add_argument(
        '--pattern-height',
        type=int,
        default=6,
        help='Số inner corners theo chiều dọc (default: 6)'
    )

    parser.add_argument(
        '--square-size',
        type=float,
        default=1.0,
        help='Kích thước 1 ô vuông (mm hoặc arbitrary, default: 1.0)'
    )

    args = parser.parse_args()

    # Setup logger
    logger = setup_logger('calibration', 'INFO', console=True)

    logger.info("=" * 60)
    logger.info("Camera Calibration Tool")
    logger.info("=" * 60)

    try:
        # Calibrate
        camera_matrix, dist_coeffs, image_size = calibrate_camera_from_images(
            args.images,
            pattern_size=(args.pattern_width, args.pattern_height),
            square_size=args.square_size
        )

        # Lưu ra file
        save_calibration_to_yaml(
            camera_matrix,
            dist_coeffs,
            image_size,
            args.output
        )

        logger.info(f"\nCalibration parameters đã lưu vào: {args.output}")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Lỗi: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
