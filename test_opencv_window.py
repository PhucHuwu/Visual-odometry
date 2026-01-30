"""Test OpenCV Window

Kiểm tra xem OpenCV có thể hiển thị cửa sổ không.
"""

import cv2
import numpy as np

# Tạo test image
img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.putText(img, "Test Window - Press 'q' to quit", (50, 240),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Hiển thị
cv2.imshow("OpenCV Test", img)
print("Cửa sổ OpenCV đã được tạo. Nhấn 'q' để thoát...")

# Wait
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
print("Test hoàn thành!")
