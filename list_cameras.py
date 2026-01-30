"""List Available Cameras

Hiển thị tất cả cameras có sẵn trên system.
"""

import cv2


def list_cameras(max_cameras=10):
    """Kiểm tra và list tất cả cameras"""
    available_cameras = []

    print("Đang kiểm tra cameras...")
    print("=" * 60)

    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)

        if cap.isOpened():
            # Get camera info
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Try to get backend name
            backend = cap.getBackendName()

            print(f"Camera ID: {i}")
            print(f"  Backend: {backend}")
            print(f"  Resolution: {int(width)}x{int(height)}")
            print(f"  FPS: {fps}")

            # Try to read a frame to verify
            ret, frame = cap.read()
            if ret:
                print(f"  Status: ✅ Working")
            else:
                print(f"  Status: ⚠️ Detected but cannot read frame")

            available_cameras.append(i)
            print()

            cap.release()

    print("=" * 60)

    if available_cameras:
        print(f"\nTìm thấy {len(available_cameras)} camera(s): {available_cameras}")
        print("\nGợi ý:")
        print(f"  - Camera ID 0: Thường là MacBook built-in camera")
        print(f"  - Camera ID 1+: Có thể là iPhone Continuity Camera hoặc external camera")
        print("\nĐể dùng camera khác, chạy:")
        print(f"  python run.py --camera 1 --algorithm orb  # Thử camera ID 1")
        print(f"  python run.py --camera 2 --algorithm orb  # Thử camera ID 2")
    else:
        print("Không tìm thấy camera nào!")

    return available_cameras


if __name__ == '__main__':
    list_cameras()
