import cv2
import numpy as np

image = cv2.imread("/Users/niteshjaiswal/.gemini/antigravity/brain/d51e66d2-1e94-4574-b124-1fe525d6d89c/.tempmediaStorage/media_d51e66d2-1e94-4574-b124-1fe525d6d89c_1778831719998.jpg")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# Black color in HSV: low Value (brightness)
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 50])
mask = cv2.inRange(hsv, lower_black, upper_black)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

print(f"Found {len(contours)} black contours.")
for i, cnt in enumerate(contours[:10]):
    area = cv2.contourArea(cnt)
    if area > 1000:
        x, y, w, h = cv2.boundingRect(cnt)
        print(f"Contour {i}: x={x}, y={y}, w={w}, h={h}, area={area}, aspect={w/h:.2f}")
