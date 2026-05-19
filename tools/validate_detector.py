import sys
import cv2
from backend.models.analyzer import ChairAnalyzer

IMAGE_PATH = "/Users/niteshjaiswal/.gemini/antigravity/brain/d51e66d2-1e94-4574-b124-1fe525d6d89c/.tempmediaStorage/media_d51e66d2-1e94-4574-b124-1fe525d6d89c_1778831719998.jpg"

img = cv2.imread(IMAGE_PATH)
if img is None:
    print(f"IMAGE_NOT_FOUND: {IMAGE_PATH}")
    sys.exit(2)

an = ChairAnalyzer()
print("Analyzer initialized. Running detection...")
dets = an.detect_objects(img)
print(f"Total detections: {len(dets)}")
chairs = [d for d in dets if getattr(d, 'class_id', None) == 56]
print(f"Chair detections (class_id==56): {len(chairs)}")
for i, c in enumerate(chairs[:20]):
    print(f"{i}: bbox={c.bbox}, center={c.center}, w={c.width}, h={c.height}, conf={getattr(c,'confidence',None)}")

if not dets:
    print("No detections returned — check that 'yolov8n.pt' exists in the project root or backend/.")
