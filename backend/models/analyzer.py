"""
AI Chair Detection & Alignment Analysis Engine.
Uses YOLOv8 (pre-trained on COCO) for object detection combined with a
custom-trained reference profile learned from the user's actual lab photos
for highly accurate chair arrangement classification.
"""

import cv2
import numpy as np
import math
import json
import os
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field, asdict

# COCO class IDs
CHAIR_CLASS_ID = 56
DESK_CLASS_ID = 60     # 'dining table' in COCO — works for desks too
BENCH_CLASS_ID = 13    # benches can be detected too

# Detection thresholds
CONFIDENCE_THRESHOLD = 0.20

# Alignment thresholds (as fraction of image dimension)
DISTANCE_THRESHOLD_RATIO = 0.05
ALIGNMENT_THRESHOLD_RATIO = 0.1
OVERLAP_MIN_RATIO = 0.15
ANGLE_DEVIATION_MAX = 25

# Trained profile path — try multiple resolution strategies for uvicorn --reload
_candidates = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_profile.json"),
    os.path.join(os.getcwd(), "models", "trained_profile.json"),
    "/Users/niteshjaiswal/Labproject/backend/models/trained_profile.json",
]
PROFILE_PATH = next((p for p in _candidates if os.path.exists(p)), _candidates[0])


@dataclass
class DetectedObject:
    """Represents a detected object (chair or desk)."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    width: int
    height: int
    area: int


@dataclass
class ChairAnalysis:
    """Analysis result for a single chair."""
    chair_id: int
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    confidence: float
    is_properly_arranged: bool
    nearest_desk_id: Optional[int]
    distance_to_desk: Optional[float]
    alignment_score: float  # 0-100
    issues: List[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Complete analysis result."""
    total_chairs: int
    total_desks: int
    correct_chairs: int
    misplaced_chairs: int
    accuracy: float
    avg_confidence: float
    chairs: List[Dict[str, Any]]
    desks: List[Dict[str, Any]]
    image_width: int
    image_height: int
    scene_classification: str = "unknown"
    scene_confidence: float = 0.0


class ChairAnalyzer:
    """
    AI-powered chair arrangement analyzer.
    Uses YOLOv8 for detection + trained reference profile for accurate
    classification on the specific lab environment.
    """

    def __init__(self):
        self.model = None
        self.trained_profile = None
        self._load_model()
        self._load_trained_profile()

    def _load_model(self):
        """Load the YOLOv8 model (auto-downloads on first run)."""
        try:
            from ultralytics import YOLO
            self.model = YOLO("yolov8n.pt")
            print("✅ YOLOv8 model loaded successfully")
        except Exception as e:
            print(f"⚠️ Failed to load YOLOv8 model: {e}")
            self.model = None

    def _load_trained_profile(self):
        """Load the trained reference profile if available."""
        if os.path.exists(PROFILE_PATH):
            try:
                with open(PROFILE_PATH, 'r') as f:
                    self.trained_profile = json.load(f)
                acc = self.trained_profile.get("training_accuracy", 0)
                samples = self.trained_profile.get("training_samples", {})
                print(f"✅ Trained profile loaded (accuracy: {acc}%, "
                      f"samples: {samples.get('correct', 0)}+{samples.get('misplaced', 0)})")
            except Exception as e:
                print(f"⚠️ Failed to load trained profile: {e}")
                self.trained_profile = None
        else:
            print("ℹ️ No trained profile found — using default thresholds")

    def detect_objects(self, image: np.ndarray) -> List[DetectedObject]:
        """Run YOLOv8 inference and extract chairs and desks."""
        if self.model is None:
            raise RuntimeError("YOLOv8 model not loaded")

        results = self.model(
            image,
            conf=CONFIDENCE_THRESHOLD,
            classes=[CHAIR_CLASS_ID, DESK_CLASS_ID, BENCH_CLASS_ID],
            verbose=False
        )

        objects = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                w = x2 - x1
                h = y2 - y1
                cx = x1 + w // 2
                cy = y1 + h // 2

                # Enforce color filter: Lab chairs are black.
                # If YOLO detects a non-black object as a chair, reject it.
                if cls_id == CHAIR_CLASS_ID:
                    # Ensure bbox is within image bounds
                    by1, by2 = max(0, y1), min(image.shape[0], y2)
                    bx1, bx2 = max(0, x1), min(image.shape[1], x2)
                    if by2 > by1 and bx2 > bx1:
                        roi = image[by1:by2, bx1:bx2]
                        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                        # Black is defined by low Value (brightness)
                        lower_black = np.array([0, 0, 0])
                        upper_black = np.array([180, 255, 60])
                        black_mask = cv2.inRange(hsv_roi, lower_black, upper_black)
                        black_ratio = cv2.countNonZero(black_mask) / (w * h)
                        if black_ratio < 0.15:
                            continue # Reject: Not enough black to be a lab chair

                class_name = self.model.names.get(cls_id, "unknown")

                objects.append(DetectedObject(
                    class_id=cls_id,
                    class_name=class_name,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    center=(cx, cy),
                    width=w,
                    height=h,
                    area=w * h
                ))

        # If no desks were found by YOLO, synthesize them using the blue partitions.
        # This gives an absolute spatial reference to evaluate if chairs are tucked in.
        has_desks = any(obj.class_id in (DESK_CLASS_ID, BENCH_CLASS_ID) for obj in objects)
        if not has_desks:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower_blue = np.array([90, 40, 80])
            upper_blue = np.array([130, 255, 255])
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            for cnt in contours[:15]:
                if cv2.contourArea(cnt) > 2000:
                    x, y, w, h = cv2.boundingRect(cnt)
                    
                    # We treat the blue partition itself as the "desk" anchor point.
                    # A tucked-in chair will overlap or be very close to it.
                    objects.append(DetectedObject(
                        class_id=DESK_CLASS_ID,
                        class_name="desk (partition)",
                        confidence=0.9,
                        bbox=(x, y, x + w, y + h),
                        center=(x + w // 2, y + h // 2),
                        width=w,
                        height=h,
                        area=w * h
                    ))

        return objects

    def _calculate_distance(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Shortest Euclidean distance between two bounding boxes. Returns 0 if they overlap."""
        x1_left, y1_top, x1_right, y1_bottom = bbox1
        x2_left, y2_top, x2_right, y2_bottom = bbox2
        
        dx = max(0, x2_left - x1_right, x1_left - x2_right)
        dy = max(0, y2_top - y1_bottom, y1_top - y2_bottom)
        
        return math.sqrt(dx**2 + dy**2)

    def _calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Calculate Intersection over Union between two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / max(union, 1)

    def _is_chair_pulled_out_of_row(self, chair: DetectedObject, all_chairs: List[DetectedObject]) -> bool:
        """
        Check if a chair is pulled out by measuring its perpendicular distance 
        from the main line of chairs. This works for both horizontal rows and 
        vertical columns (aisle setups).
        """
        if len(all_chairs) < 3:
            return False
            
        # Extract all center points
        points = np.array([c.center for c in all_chairs])
        
        # Fit a line (PCA/SVD) to find the main axis of the chairs
        mean = np.mean(points, axis=0)
        centered = points - mean
        
        if len(centered) == 0 or np.all(centered == 0):
             return False
             
        U, S, Vt = np.linalg.svd(centered)
        # The main axis is the first row of Vt
        main_axis = Vt[0]
        # The normal vector is perpendicular to the main axis
        normal = np.array([-main_axis[1], main_axis[0]])
        
        # Calculate the median distance along the normal vector for all chairs
        distances = [np.dot((p - mean), normal) for p in points]
        median_dist = np.median(distances)
        
        # Calculate this chair's distance
        chair_dist = np.dot((np.array(chair.center) - mean), normal)
        
        # Deviation from the median line
        deviation = abs(chair_dist - median_dist)
        
        # Threshold: if it protrudes by more than ~25% of its width
        is_pulled_out = deviation > chair.width * 0.25
        
        return is_pulled_out

    def _estimate_tilt_angle(self, obj: DetectedObject) -> float:
        """
        Estimate tilt angle from bounding box aspect ratio.
        In this lab, tucked-in chairs are viewed from the side (narrow bounding box, aspect ~0.5-0.6).
        When a chair is rotated to face the aisle/camera, it becomes wider (aspect > 0.75).
        """
        if obj.height == 0:
            return 0.0
        aspect_ratio = obj.width / obj.height
        
        # If it's wider than normal, it's rotated towards the camera
        if aspect_ratio > 0.8:
            return 45.0
        elif aspect_ratio > 0.75:
            return 30.0
        return 0.0

    def _extract_scene_features(self, image: np.ndarray, chairs, desks) -> Dict[str, float]:
        """
        Extract the same scene-level features used during training.
        This allows comparison against the trained profile.
        """
        h, w = image.shape[:2]
        img_area = h * w
        img_diagonal = np.sqrt(w**2 + h**2)

        chair_data = []
        for c in chairs:
            bw = c.width
            bh = c.height
            chair_data.append({
                "bbox": list(c.bbox), "center": list(c.center),
                "width": bw, "height": bh, "area": c.area,
                "confidence": c.confidence,
                "aspect_ratio": bh / max(bw, 1)
            })

        desk_data = []
        for d in desks:
            desk_data.append({
                "bbox": list(d.bbox), "center": list(d.center),
                "width": d.width, "height": d.height, "area": d.area,
            })

        num_chairs = len(chair_data)
        num_desks = len(desk_data)

        chair_areas = [c["area"] / img_area for c in chair_data] if chair_data else [0]
        chair_y_positions = [c["center"][1] / h for c in chair_data] if chair_data else [0]
        chair_x_positions = [c["center"][0] / w for c in chair_data] if chair_data else [0]
        chair_aspect_ratios = [c["aspect_ratio"] for c in chair_data] if chair_data else [0]

        chair_spacings = []
        if len(chair_data) >= 2:
            sorted_chairs = sorted(chair_data, key=lambda c: c["center"][1])
            for i in range(1, len(sorted_chairs)):
                dist = np.sqrt(
                    (sorted_chairs[i]["center"][0] - sorted_chairs[i-1]["center"][0])**2 +
                    (sorted_chairs[i]["center"][1] - sorted_chairs[i-1]["center"][1])**2
                ) / img_diagonal
                chair_spacings.append(dist)

        overlap_ratios = []
        chair_desk_distances = []
        for chair in chair_data:
            best_overlap = 0
            best_dist = float('inf')
            for desk in desk_data:
                ix1 = max(chair["bbox"][0], desk["bbox"][0])
                iy1 = max(chair["bbox"][1], desk["bbox"][1])
                ix2 = min(chair["bbox"][2], desk["bbox"][2])
                iy2 = min(chair["bbox"][3], desk["bbox"][3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                union = chair["area"] + desk["area"] - inter
                iou = inter / max(union, 1)
                best_overlap = max(best_overlap, iou)
                dist = np.sqrt(
                    (chair["center"][0] - desk["center"][0])**2 +
                    (chair["center"][1] - desk["center"][1])**2
                ) / img_diagonal
                best_dist = min(best_dist, dist)
            overlap_ratios.append(best_overlap)
            chair_desk_distances.append(best_dist if best_dist != float('inf') else 1.0)

        # Floor exposure analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lower_floor = np.array([0, 0, 150])
        upper_floor = np.array([180, 60, 255])
        floor_mask = cv2.inRange(hsv, lower_floor, upper_floor)
        lower_half = floor_mask[h//2:, :]
        floor_exposure = np.sum(lower_half > 0) / max(lower_half.size, 1)

        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        lower_edges = edges[h//2:, :]
        edge_density = np.sum(lower_edges > 0) / max(lower_edges.size, 1)

        # Structural regularity
        x_variance = float(np.var(chair_x_positions)) if len(chair_x_positions) > 1 else 0
        y_variance = float(np.var(chair_y_positions)) if len(chair_y_positions) > 1 else 0
        aspect_variance = float(np.var(chair_aspect_ratios)) if len(chair_aspect_ratios) > 1 else 0

        # Blue desk divider exposure
        lower_blue = np.array([90, 40, 80])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_exposure = np.sum(blue_mask > 0) / max(blue_mask.size, 1)

        return {
            "num_chairs": float(num_chairs),
            "num_desks": float(num_desks),
            "avg_chair_area": float(np.mean(chair_areas)),
            "std_chair_area": float(np.std(chair_areas)),
            "avg_chair_y": float(np.mean(chair_y_positions)),
            "std_chair_y": float(np.std(chair_y_positions)) if len(chair_y_positions) > 1 else 0,
            "avg_chair_x": float(np.mean(chair_x_positions)),
            "std_chair_x": float(np.std(chair_x_positions)) if len(chair_x_positions) > 1 else 0,
            "avg_aspect_ratio": float(np.mean(chair_aspect_ratios)),
            "aspect_variance": float(aspect_variance),
            "avg_spacing": float(np.mean(chair_spacings)) if chair_spacings else 0,
            "std_spacing": float(np.std(chair_spacings)) if len(chair_spacings) > 1 else 0,
            "avg_desk_overlap": float(np.mean(overlap_ratios)) if overlap_ratios else 0,
            "min_desk_overlap": float(np.min(overlap_ratios)) if overlap_ratios else 0,
            "avg_desk_distance": float(np.mean(chair_desk_distances)) if chair_desk_distances else 1.0,
            "max_desk_distance": float(np.max(chair_desk_distances)) if chair_desk_distances else 1.0,
            "floor_exposure": float(floor_exposure),
            "edge_density": float(edge_density),
            "x_variance": float(x_variance),
            "y_variance": float(y_variance),
            "blue_desk_exposure": float(blue_exposure),
        }

    def _classify_scene(self, features: Dict[str, float]) -> Tuple[str, float]:
        """
        Classify the entire scene using the trained profile classifier.
        Returns (classification, confidence) where classification is 'correct' or 'misplaced'.
        """
        if not self.trained_profile:
            return "unknown", 0.0

        profile = self.trained_profile
        feature_keys = profile["feature_keys"]
        norm_mean = np.array(profile["normalization"]["mean"])
        norm_std = np.array(profile["normalization"]["std"])
        decision_boundary = np.array(profile["classifier"]["decision_boundary"])
        decision_direction = np.array(profile["classifier"]["decision_direction"])

        # Build feature vector in same order as training
        feat_vec = np.array([features.get(k, 0.0) for k in feature_keys])
        feat_norm = (feat_vec - norm_mean) / norm_std

        # Project onto decision direction
        projection = np.dot(feat_norm - decision_boundary, decision_direction)

        # Convert to confidence (sigmoid-like mapping)
        confidence = 1.0 / (1.0 + np.exp(-abs(projection) * 3))
        confidence = min(max(confidence, 0.5), 0.99)

        # Note: decision_direction points from correct→misplaced centroid,
        # so positive projection means closer to misplaced
        if projection > 0:
            return "correct", float(confidence)
        else:
            return "misplaced", float(confidence)

    def analyze_arrangement(self, image: np.ndarray) -> AnalysisResult:
        """
        Full analysis pipeline combining YOLO detection with trained profile:
        1. Detect all chairs and desks using YOLO
        2. Extract scene-level features
        3. Classify scene using trained profile (if available)
        4. Per-chair analysis using spatial rules refined by training data
        5. Combine scene + per-chair results for final classification
        """
        h, w = image.shape[:2]
        image_diagonal = math.sqrt(w ** 2 + h ** 2)

        # Detect objects
        objects = self.detect_objects(image)
        chairs = [o for o in objects if o.class_id == CHAIR_CLASS_ID]
        desks = [o for o in objects if o.class_id in (DESK_CLASS_ID, BENCH_CLASS_ID)]

        # Scene classification using trained profile
        scene_class = "unknown"
        scene_conf = 0.0
        if self.trained_profile and chairs:
            scene_features = self._extract_scene_features(image, chairs, desks)
            scene_class, scene_conf = self._classify_scene(scene_features)
            print(f"  🔍 Scene classification: {scene_class} (confidence: {scene_conf:.1%})")

        # Get trained thresholds if available
        if self.trained_profile and "thresholds" in self.trained_profile:
            thresholds = self.trained_profile["thresholds"]
        else:
            thresholds = None

        # Dynamic thresholds based on image size
        max_distance = image_diagonal * DISTANCE_THRESHOLD_RATIO
        max_alignment_dev = w * ALIGNMENT_THRESHOLD_RATIO

        # Analyze each chair
        chair_analyses = []
        for idx, chair in enumerate(chairs):
            issues = []
            nearest_desk_id = None
            nearest_distance = float('inf')
            alignment_score = 100.0

            # Find nearest desk
            for didx, desk in enumerate(desks):
                dist = self._calculate_distance(chair.bbox, desk.bbox)
                if dist < nearest_distance:
                    nearest_distance = dist
                    nearest_desk_id = didx

            is_properly_arranged = True

            if nearest_desk_id is not None:
                desk = desks[nearest_desk_id]

                # Removed distance check (Check 1) per user request to ONLY mark rotated chairs
                # Removed Check 2 & 3
                
                # Check 4: Tilt/rotation
                tilt = self._estimate_tilt_angle(chair)
                if tilt > 0:
                    is_properly_arranged = False
                    issues.append(f"Tilted/rotated (~{tilt}°)")
                    alignment_score -= 30

                # Removed row alignment check (Check 5) per user request
            else:
                nearest_distance = None
                # Fallback
                tilt = self._estimate_tilt_angle(chair)
                if tilt > 0:
                    is_properly_arranged = False
                    issues.append(f"Tilted/rotated (~{tilt}°)")
                    alignment_score -= 30
                else:
                    is_properly_arranged = True
                    alignment_score = 90.0
            # --- Apply trained profile scene-level override ---
            # Use the scene classification merely as a hint to adjust the score,
            # but rely primarily on the spatial rules (which now work thanks to the synthesized desks).
            if scene_class != "unknown" and scene_conf > 0.5:
                if scene_class == "correct":
                    alignment_score = min(100, alignment_score * 1.15)
                    # Only override if it's borderline
                    if not is_properly_arranged and len(issues) <= 1 and alignment_score > 70:
                        is_properly_arranged = True
                        issues = ["Minor deviation (within acceptable range)"]
                elif scene_class == "misplaced":
                    alignment_score *= 0.85
                    if is_properly_arranged and alignment_score < 70:
                        is_properly_arranged = False
                        issues.append("Scene context suggests misalignment")

            alignment_score = max(0, min(100, alignment_score))

            chair_analyses.append(ChairAnalysis(
                chair_id=idx,
                bbox=chair.bbox,
                center=chair.center,
                confidence=chair.confidence,
                is_properly_arranged=is_properly_arranged,
                nearest_desk_id=nearest_desk_id,
                distance_to_desk=nearest_distance if nearest_distance != float('inf') else None,
                alignment_score=alignment_score,
                issues=issues
            ))

        # Calculate totals
        correct = sum(1 for c in chair_analyses if c.is_properly_arranged)
        misplaced = len(chair_analyses) - correct
        accuracy = (correct / max(len(chair_analyses), 1)) * 100
        avg_conf = (sum(c.confidence for c in chair_analyses) / max(len(chair_analyses), 1)) * 100

        return AnalysisResult(
            total_chairs=len(chairs),
            total_desks=len(desks),
            correct_chairs=correct,
            misplaced_chairs=misplaced,
            accuracy=round(accuracy, 1),
            avg_confidence=round(avg_conf, 1),
            chairs=[asdict(c) for c in chair_analyses],
            desks=[{
                "desk_id": i,
                "bbox": d.bbox,
                "center": d.center,
                "confidence": d.confidence,
                "class_name": d.class_name
            } for i, d in enumerate(desks)],
            image_width=w,
            image_height=h,
            scene_classification=scene_class,
            scene_confidence=round(scene_conf * 100, 1)
        )

    def annotate_image(self, image: np.ndarray, result: AnalysisResult) -> np.ndarray:
        """
        Draw bounding boxes and labels on the image:
        - GREEN for properly arranged chairs
        - RED for misplaced chairs
        - BLUE for desks
        """
        annotated = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2

        # Draw desks (blue)
        for desk in result.desks:
            x1, y1, x2, y2 = desk["bbox"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 180, 50), 2)
            label = f"Desk ({desk['confidence']:.0%})"
            label_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), (x1 + label_size[0] + 6, y1), (255, 180, 50), -1)
            cv2.putText(annotated, label, (x1 + 3, y1 - 5), font, font_scale, (0, 0, 0), thickness)

        # Draw chairs
        for chair in result.chairs:
            x1, y1, x2, y2 = chair["bbox"]
            is_ok = chair["is_properly_arranged"]

            if is_ok:
                color = (0, 220, 80)   # Green
                status_mark = "OK"
            else:
                color = (0, 0, 255)    # Red
                status_mark = "X"

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

            # Draw label
            label = f"Chair #{chair['chair_id']+1} {status_mark} ({chair['confidence']:.0%})"
            label_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), (x1 + label_size[0] + 6, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 3, y1 - 5), font, font_scale, (255, 255, 255), thickness)

            # Draw issues for misplaced chairs
            if not is_ok and chair["issues"]:
                for i, issue in enumerate(chair["issues"][:3]):
                    issue_y = y2 + 18 + (i * 18)
                    cv2.putText(annotated, f"! {issue}", (x1, issue_y), font, 0.4, (0, 0, 255), 1)

            # Draw alignment score bar
            score = chair["alignment_score"]
            bar_x = x1
            bar_y = y2 + 5
            bar_width = x2 - x1
            bar_height = 6
            filled_width = int(bar_width * score / 100)

            cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (60, 60, 60), -1)
            bar_color = (0, 220, 80) if score >= 70 else (0, 180, 255) if score >= 40 else (0, 0, 255)
            cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), bar_color, -1)

        # Draw summary overlay
        summary_h = 110 if result.scene_classification != "unknown" else 90
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (380, summary_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

        cv2.putText(annotated, f"Chairs: {result.total_chairs} | Desks: {result.total_desks}",
                    (10, 25), font, 0.6, (255, 255, 255), 1)
        cv2.putText(annotated, f"Correct: {result.correct_chairs} | Misplaced: {result.misplaced_chairs}",
                    (10, 50), font, 0.6, (255, 255, 255), 1)

        acc_color = (0, 220, 80) if result.accuracy >= 80 else (0, 180, 255) if result.accuracy >= 50 else (0, 0, 255)
        cv2.putText(annotated, f"Accuracy: {result.accuracy}%",
                    (10, 75), font, 0.7, acc_color, 2)

        if result.scene_classification != "unknown":
            sc_color = (0, 220, 80) if result.scene_classification == "correct" else (0, 0, 255)
            sc_text = f"AI Scene: {result.scene_classification.upper()} ({result.scene_confidence}%)"
            cv2.putText(annotated, sc_text, (10, 100), font, 0.5, sc_color, 1)

        return annotated

    def generate_heatmap(self, image: np.ndarray, result: AnalysisResult) -> np.ndarray:
        """Generate a heatmap overlay showing problematic areas."""
        h, w = image.shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)

        for chair in result.chairs:
            if not chair["is_properly_arranged"]:
                x1, y1, x2, y2 = chair["bbox"]
                cx, cy = chair["center"]
                radius = max(x2 - x1, y2 - y1)
                y_grid, x_grid = np.ogrid[max(0, cy-radius):min(h, cy+radius),
                                           max(0, cx-radius):min(w, cx+radius)]
                mask = ((x_grid - cx)**2 + (y_grid - cy)**2) <= radius**2
                if mask.any():
                    blob_h, blob_w = mask.shape
                    start_y = max(0, cy - radius)
                    start_x = max(0, cx - radius)
                    heatmap[start_y:start_y+blob_h, start_x:start_x+blob_w] += mask.astype(np.float32)

        if heatmap.max() > 0:
            heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            blended = cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)
            return blended
        return image


# Global analyzer instance (singleton)
_analyzer_instance = None


def get_analyzer() -> ChairAnalyzer:
    """Get or create the global analyzer instance."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = ChairAnalyzer()
    return _analyzer_instance


def reload_analyzer():
    """Force reload the analyzer (e.g. after training a new profile)."""
    global _analyzer_instance
    _analyzer_instance = ChairAnalyzer()
    return _analyzer_instance
