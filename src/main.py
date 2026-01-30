"""Main Entry Point

Chạy Visual Odometry system.
"""

from utils.logger import setup_logger
from utils.config_loader import ConfigLoader
from core.vo_pipeline import VOPipeline
from core.camera import CameraInput
from visualization.visualizer import Visualizer
import argparse
import sys
import time
from pathlib import Path

# Add src to path
src_path = Path(__file__).resolve().parent
sys.path.insert(0, str(src_path))


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Visual Odometry System')

    parser.add_argument(
        '--video',
        type=str,
        default=None,
        help='Path tới video file (None = sử dụng webcam)'
    )

    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera device ID (default: 0)'
    )

    parser.add_argument(
        '--algorithm',
        type=str,
        choices=['fast', 'orb', 'sift', 'lucas_kanade'],
        default='orb',
        help='Algorithm để sử dụng (default: orb)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path tới config file (default: config/default_config.yaml)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='output/trajectory.txt',
        help='Output trajectory file (default: output/trajectory.txt)'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Log level (default: INFO)'
    )

    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Tắt hiển thị camera window (chỉ process, không show)'
    )

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    # Setup logger
    logger = setup_logger(
        name='vo',
        level=args.log_level,
        log_file='logs/vo.log',
        console=True
    )

    logger.info("=" * 60)
    logger.info("Visual Odometry System Started")
    logger.info("=" * 60)

    try:
        # Load config
        config = ConfigLoader(args.config)

        # Override algorithm từ command line
        config.set('algorithm.type', args.algorithm)

        # Determine video source
        if args.video is not None:
            source = args.video
            logger.info(f"Sử dụng video file: {source}")
        else:
            source = args.camera
            logger.info(f"Sử dụng webcam: {source}")

        # Initialize camera
        camera = CameraInput(
            source=source,
            width=config.get('video.width'),
            height=config.get('video.height')
        )

        # Initialize VO pipeline
        vo = VOPipeline(config)

        # Initialize visualizer
        visualizer = None
        if not args.no_display:
            visualizer = Visualizer(window_name=f"VO - {args.algorithm.upper()}")
            logger.info("Visualization enabled - cửa sổ camera sẽ hiện")

        logger.info(f"Bắt đầu xử lý video với algorithm: {args.algorithm.upper()}")
        logger.info("Nhấn 'q' để thoát")

        frame_times = []

        # Main loop
        while True:
            start_time = time.time()

            # Read frame
            ret, frame = camera.read()

            if not ret:
                logger.info("Hết video hoặc camera bị ngắt kết nối")
                break

            # Process frame
            success = vo.process_frame(frame)

            # Calculate FPS
            elapsed = time.time() - start_time
            frame_times.append(elapsed)

            # Giữ lại 30 frames gần nhất để tính FPS
            if len(frame_times) > 30:
                frame_times.pop(0)

            avg_time = sum(frame_times) / len(frame_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0.0

            # Visualize nếu enabled
            if visualizer is not None:
                # Get stats
                stats = vo.get_stats()
                stats['fps'] = fps
                stats['keypoints'] = len(vo.prev_keypoints) if vo.prev_keypoints else 0
                stats['algorithm'] = args.algorithm.upper()

                # Draw frame
                display_frame = frame.copy()

                # Draw keypoints nếu có
                if vo.prev_keypoints:
                    display_frame = visualizer.draw_keypoints(display_frame, vo.prev_keypoints)

                # Add stats overlay
                display_frame = visualizer.add_stats(display_frame, stats)

                # Show
                if not visualizer.show(display_frame, wait_ms=1):
                    logger.info("User nhấn 'q' để thoát")
                    break

            # Log progress định kỳ
            if camera.get_frame_count() % 10 == 0:
                stats = vo.get_stats()
                logger.info(f"Frame {camera.get_frame_count()}: "
                            f"FPS={fps:.1f}, "
                            f"Trajectory={stats['trajectory_length']} poses, "
                            f"Position={stats['current_position']}")

        # Lưu trajectory
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        vo.save_trajectory(str(output_path))

        # Final stats
        logger.info("=" * 60)
        logger.info("VO Pipeline Finished!")

        stats = vo.get_stats()
        logger.info(f"Tổng frames xử lý: {stats['frame_count']}")
        logger.info(f"Trajectory length: {stats['trajectory_length']} poses")
        logger.info(f"Average keypoints/frame: {stats['avg_keypoints_per_frame']:.1f}")
        logger.info(f"Final position: {stats['current_position']}")
        logger.info(f"Trajectory đã lưu vào: {output_path}")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.info("\nNgắt bởi user (Ctrl+C)")

    except Exception as e:
        logger.error(f"Lỗi: {e}", exc_info=True)
        return 1

    finally:
        # Cleanup
        if 'visualizer' in locals() and visualizer is not None:
            visualizer.close()
        if 'camera' in locals():
            camera.release()

    return 0


if __name__ == '__main__':
    sys.exit(main())
