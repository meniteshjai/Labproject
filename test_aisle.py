import cv2
from backend.models.analyzer import ChairAnalyzer

analyzer = ChairAnalyzer()
image = cv2.imread("/Users/niteshjaiswal/.gemini/antigravity/brain/d51e66d2-1e94-4574-b124-1fe525d6d89c/.tempmediaStorage/media_d51e66d2-1e94-4574-b124-1fe525d6d89c_1778831719998.jpg")
objects = analyzer.detect_objects(image)
chairs = [o for o in objects if o.class_id == 56]
for c in chairs:
    tilt = analyzer._estimate_tilt_angle(c)
    pulled_out = analyzer._is_chair_pulled_out_of_row(c, chairs)
    print(f"Chair at y={c.center[1]}: tilt={tilt:.1f}, pulled_out={pulled_out}")
