import cv2
import numpy as np
from backend.models.analyzer import ChairAnalyzer

analyzer = ChairAnalyzer()
image = cv2.imread("/Users/niteshjaiswal/.gemini/antigravity/brain/d51e66d2-1e94-4574-b124-1fe525d6d89c/.tempmediaStorage/media_d51e66d2-1e94-4574-b124-1fe525d6d89c_1778831719998.jpg")
objects = analyzer.detect_objects(image)
chairs = [o for o in objects if o.class_id == 56]

for i, c in enumerate(chairs):
    roi = image[c.bbox[1]:c.bbox[3], c.bbox[0]:c.bbox[2]]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 60])
    black_mask = cv2.inRange(hsv_roi, lower_black, upper_black)
    black_ratio = cv2.countNonZero(black_mask) / (c.width * c.height)
    print(f"Chair {i} at y={c.center[1]}: black_ratio={black_ratio:.2f}")
