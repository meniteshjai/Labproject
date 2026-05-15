"""
Training script for Smart Lab Chair Monitoring System.
Learns reference patterns from labeled lab photos (correct vs misplaced).

Approach:
1. Use YOLOv8 to detect chairs in each labeled image
2. Extract spatial features: chair positions, desk-chair overlap, spacing, floor exposure
3. Use OpenCV to extract structural features: edge density, color histograms, contour patterns
4. Build a reference profile from "correct" images
5. Train a feature-based classifier (SVM) for scene-level classification
6. Save trained profile + classifier for use in the analyzer
"""

import os
import sys
import json
import cv2
import numpy as np
import shutil
from datetime import datetime

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO

# ===== CONFIGURATION =====
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CORRECT_DIR = os.path.join(DATA_DIR, "correct")
MISPLACED_DIR = os.path.join(DATA_DIR, "misplaced")
PROFILE_PATH = os.path.join(PROJECT_ROOT, "models", "trained_profile.json")

CHAIR_CLASS_ID = 56
DESK_CLASS_ID = 60

# Source images from user's lab photos (in Labproject root)
LAB_PHOTOS_DIR = os.path.dirname(PROJECT_ROOT)


def organize_training_data():
    """
    Organize user's lab photos into correct/misplaced directories.
    Based on user's labeling of their uploaded photos.
    """
    os.makedirs(CORRECT_DIR, exist_ok=True)
    os.makedirs(MISPLACED_DIR, exist_ok=True)

    # User-provided labels based on their feedback
    labeled_images = {
        # CORRECT: chairs properly tucked into desks
        "correct": [
            "Unknown.jpeg",
            "WhatsApp Image 2026-05-15 at 1.14.52 PM.jpeg",
            "WhatsApp Image 2026-05-15 at 1.15.00 PM.jpeg",
            "WhatsApp Image 2026-05-15 at 1.15.07 PM.jpeg",
        ],
        # MISPLACED: chairs pulled out, scattered, not tucked in
        "misplaced": [
            "Unknown-1.jpeg",
            "Unknown-2.jpeg",
            "WhatsApp Image 2026-05-15 at 1.14.53 PM.jpeg",
            "WhatsApp Image 2026-05-15 at 1.14.53 PM-2.jpeg",
            "WhatsApp Image 2026-05-15 at 1.15.03 PM.jpeg",
            "Rotated_Chair_Sample.jpg"
        ],
    }

    # Do not wipe existing directories. Just organize new ones if needed.
    # The user can drop images directly into these folders to train.
    copied = {"correct": 0, "misplaced": 0}
    for label, filenames in labeled_images.items():
        dest_dir = CORRECT_DIR if label == "correct" else MISPLACED_DIR
        for fname in filenames:
            src = os.path.join(LAB_PHOTOS_DIR, fname)
            dst = os.path.join(dest_dir, fname)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
                copied[label] += 1
                print(f"  ✅ [{label.upper()}] {fname}")
            elif not os.path.exists(src) and not os.path.exists(dst):
                print(f"  ⚠️  Not found: {fname}")

    print(f"\n📁 Organized: {copied['correct']} correct, {copied['misplaced']} misplaced")
    return copied


def extract_features(image_path, model):
    """
    Extract comprehensive features from a lab image for training.
    Returns a feature dictionary describing the chair arrangement state.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None

    h, w = image.shape[:2]
    img_area = h * w
    img_diagonal = np.sqrt(w**2 + h**2)

    # --- 1. YOLO Detection ---
    results = model(image, conf=0.2, classes=[CHAIR_CLASS_ID, DESK_CLASS_ID], verbose=False)
    chairs = []
    desks = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            bw = x2 - x1
            bh = y2 - y1
            obj = {
                "bbox": [x1, y1, x2, y2], "center": [cx, cy],
                "width": bw, "height": bh, "area": bw * bh,
                "confidence": conf, "aspect_ratio": bh / max(bw, 1)
            }
            if cls_id == CHAIR_CLASS_ID:
                # Color filter: Chairs are black. Reject false positives.
                by1, by2 = max(0, y1), min(image.shape[0], y2)
                bx1, bx2 = max(0, x1), min(image.shape[1], x2)
                if by2 > by1 and bx2 > bx1:
                    roi = image[by1:by2, bx1:bx2]
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    lower_black = np.array([0, 0, 0])
                    upper_black = np.array([180, 255, 60])
                    black_mask = cv2.inRange(hsv_roi, lower_black, upper_black)
                    black_ratio = cv2.countNonZero(black_mask) / (bw * bh)
                    if black_ratio < 0.15:
                        continue
                chairs.append(obj)
            elif cls_id == DESK_CLASS_ID:
                desks.append(obj)

    num_chairs = len(chairs)
    num_desks = len(desks)

    # --- 2. Chair Spatial Features ---
    chair_areas = [c["area"] / img_area for c in chairs] if chairs else [0]
    chair_y_positions = [c["center"][1] / h for c in chairs] if chairs else [0]
    chair_x_positions = [c["center"][0] / w for c in chairs] if chairs else [0]
    chair_aspect_ratios = [c["aspect_ratio"] for c in chairs] if chairs else [0]

    # Distance between consecutive chairs (if sorted by position)
    chair_spacings = []
    if len(chairs) >= 2:
        sorted_chairs = sorted(chairs, key=lambda c: c["center"][1])
        for i in range(1, len(sorted_chairs)):
            dist = np.sqrt(
                (sorted_chairs[i]["center"][0] - sorted_chairs[i-1]["center"][0])**2 +
                (sorted_chairs[i]["center"][1] - sorted_chairs[i-1]["center"][1])**2
            ) / img_diagonal
            chair_spacings.append(dist)

    # Chair-desk overlap analysis
    overlap_ratios = []
    chair_desk_distances = []
    for chair in chairs:
        best_overlap = 0
        best_dist = float('inf')
        for desk in desks:
            # IoU
            ix1 = max(chair["bbox"][0], desk["bbox"][0])
            iy1 = max(chair["bbox"][1], desk["bbox"][1])
            ix2 = min(chair["bbox"][2], desk["bbox"][2])
            iy2 = min(chair["bbox"][3], desk["bbox"][3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = chair["area"] + desk["area"] - inter
            iou = inter / max(union, 1)
            best_overlap = max(best_overlap, iou)
            # Distance
            dist = np.sqrt(
                (chair["center"][0] - desk["center"][0])**2 +
                (chair["center"][1] - desk["center"][1])**2
            ) / img_diagonal
            best_dist = min(best_dist, dist)
        overlap_ratios.append(best_overlap)
        chair_desk_distances.append(best_dist if best_dist != float('inf') else 1.0)

    # --- 3. Floor exposure analysis (chairs pulled out = more visible floor) ---
    # Convert to HSV, look for floor-colored pixels in chair regions
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Floor is typically light colored (high value, low saturation)
    lower_floor = np.array([0, 0, 150])
    upper_floor = np.array([180, 60, 255])
    floor_mask = cv2.inRange(hsv, lower_floor, upper_floor)

    # Analyze lower half of image (where chairs/floor are)
    lower_half = floor_mask[h//2:, :]
    floor_exposure = np.sum(lower_half > 0) / max(lower_half.size, 1)

    # --- 4. Edge density in chair regions (misplaced chairs create irregular edge patterns) ---
    edges = cv2.Canny(gray, 50, 150)
    lower_edges = edges[h//2:, :]
    edge_density = np.sum(lower_edges > 0) / max(lower_edges.size, 1)

    # --- 5. Structural regularity (chair alignment variance) ---
    x_variance = np.var(chair_x_positions) if len(chair_x_positions) > 1 else 0
    y_variance = np.var(chair_y_positions) if len(chair_y_positions) > 1 else 0
    aspect_variance = np.var(chair_aspect_ratios) if len(chair_aspect_ratios) > 1 else 0

    # --- 6. Color histogram features (desk region vs chair region) ---
    # The blue desk dividers are distinctive - their visibility indicates chair arrangement
    lower_blue = np.array([90, 40, 80])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_exposure = np.sum(blue_mask > 0) / max(blue_mask.size, 1)

    # --- Build feature vector ---
    # We deliberately exclude scale/zoom-dependent features like num_chairs and avg_chair_area
    # because they confuse the classifier across different camera angles.
    features = {
        "std_chair_y": float(np.std(chair_y_positions)) if len(chair_y_positions) > 1 else 0,
        "std_chair_x": float(np.std(chair_x_positions)) if len(chair_x_positions) > 1 else 0,
        "avg_aspect_ratio": float(np.mean(chair_aspect_ratios)) if chair_aspect_ratios else 0,
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

    return features


def get_feature_vector(features):
    """Convert feature dict to numpy array in consistent order."""
    keys = sorted(features.keys())
    return np.array([features[k] for k in keys]), keys


def train():
    """Main training pipeline."""
    print("=" * 60)
    print("🧠 SMART LAB CHAIR MONITORING — TRAINING PIPELINE")
    print("=" * 60)

    # Step 1: Organize data
    print("\n📂 Step 1: Organizing training data...")
    organize_training_data()
    
    correct_count = len([f for f in os.listdir(CORRECT_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))])
    misplaced_count = len([f for f in os.listdir(MISPLACED_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))])

    if correct_count == 0 and misplaced_count == 0:
        print("❌ No training images found in data directories!")
        return

    # Step 2: Load YOLO model
    print("\n🤖 Step 2: Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt")
    print("  ✅ Model loaded")

    # Step 3: Extract features
    print("\n🔍 Step 3: Extracting features from training images...")
    correct_features = []
    misplaced_features = []

    for fname in os.listdir(CORRECT_DIR):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(CORRECT_DIR, fname)
            print(f"  Processing [CORRECT] {fname}...")
            feats = extract_features(path, model)
            if feats:
                correct_features.append(feats)
                print(f"    → Features extracted (Floor: {feats['floor_exposure']:.3f}, Overlap: {feats['avg_desk_overlap']:.3f})")

    for fname in os.listdir(MISPLACED_DIR):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(MISPLACED_DIR, fname)
            print(f"  Processing [MISPLACED] {fname}...")
            feats = extract_features(path, model)
            if feats:
                misplaced_features.append(feats)
                print(f"    → Features extracted (Floor: {feats['floor_exposure']:.3f}, Overlap: {feats['avg_desk_overlap']:.3f})")

    if not correct_features or not misplaced_features:
        print("❌ Need at least 1 correct and 1 misplaced image for training!")
        return

    # Step 4: Build reference profile
    print("\n📊 Step 4: Building reference profile...")

    # Average features for each class
    def avg_features(feature_list):
        keys = feature_list[0].keys()
        return {k: float(np.mean([f[k] for f in feature_list])) for k in keys}

    def std_features(feature_list):
        keys = feature_list[0].keys()
        return {k: float(np.std([f[k] for f in feature_list])) for k in keys}

    correct_avg = avg_features(correct_features)
    correct_std = std_features(correct_features)
    misplaced_avg = avg_features(misplaced_features)
    misplaced_std = std_features(misplaced_features)

    # Step 5: Train SVM classifier
    print("\n🧮 Step 5: Training classifier...")
    X = []
    y = []
    feature_keys = sorted(correct_features[0].keys())

    for f in correct_features:
        X.append([f[k] for k in feature_keys])
        y.append(0)  # 0 = correct
    for f in misplaced_features:
        X.append([f[k] for k in feature_keys])
        y.append(1)  # 1 = misplaced

    X = np.array(X)
    y = np.array(y)

    # Normalize features
    feat_mean = X.mean(axis=0)
    feat_std = X.std(axis=0) + 1e-8
    X_norm = (X - feat_mean) / feat_std

    # Simple linear classifier weights (since small dataset, use centroid-based)
    correct_centroid = X_norm[y == 0].mean(axis=0)
    misplaced_centroid = X_norm[y == 1].mean(axis=0)

    # Decision boundary = midpoint between centroids
    decision_boundary = (correct_centroid + misplaced_centroid) / 2
    decision_direction = misplaced_centroid - correct_centroid
    decision_direction = decision_direction / (np.linalg.norm(decision_direction) + 1e-8)

    # Test on training data
    correct_count = 0
    for i, x in enumerate(X_norm):
        projection = np.dot(x - decision_boundary, decision_direction)
        predicted = 1 if projection > 0 else 0
        actual = y[i]
        if predicted == actual:
            correct_count += 1

    train_accuracy = correct_count / len(y) * 100
    print(f"  ✅ Training accuracy: {train_accuracy:.1f}%")

    # Step 6: Compute discriminative thresholds
    print("\n📏 Step 6: Computing arrangement thresholds...")

    # Find the most discriminative features
    feature_importance = {}
    for i, key in enumerate(feature_keys):
        correct_vals = X[y == 0, i]
        misplaced_vals = X[y == 1, i]
        separation = abs(correct_vals.mean() - misplaced_vals.mean())
        pooled_std = np.sqrt((correct_vals.std()**2 + misplaced_vals.std()**2) / 2) + 1e-8
        fisher_score = separation / pooled_std
        feature_importance[key] = {
            "fisher_score": float(fisher_score),
            "correct_mean": float(correct_vals.mean()),
            "correct_std": float(correct_vals.std()),
            "misplaced_mean": float(misplaced_vals.mean()),
            "misplaced_std": float(misplaced_vals.std()),
            "threshold": float((correct_vals.mean() + misplaced_vals.mean()) / 2),
            "direction": "higher_is_misplaced" if misplaced_vals.mean() > correct_vals.mean() else "lower_is_misplaced"
        }

    # Sort by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1]["fisher_score"], reverse=True)

    print("\n  Top discriminative features:")
    for fname, finfo in sorted_features[:8]:
        print(f"    {fname}: score={finfo['fisher_score']:.3f} "
              f"(correct={finfo['correct_mean']:.4f}, misplaced={finfo['misplaced_mean']:.4f})")

    # Step 7: Save trained profile
    print("\n💾 Step 7: Saving trained profile...")

    profile = {
        "version": "1.0",
        "trained_at": datetime.now().isoformat(),
        "training_samples": {
            "correct": len(correct_features),
            "misplaced": len(misplaced_features)
        },
        "training_accuracy": train_accuracy,
        "feature_keys": feature_keys,
        "normalization": {
            "mean": feat_mean.tolist(),
            "std": feat_std.tolist()
        },
        "classifier": {
            "type": "centroid_linear",
            "correct_centroid": correct_centroid.tolist(),
            "misplaced_centroid": misplaced_centroid.tolist(),
            "decision_boundary": decision_boundary.tolist(),
            "decision_direction": decision_direction.tolist()
        },
        "reference_profiles": {
            "correct": {
                "avg": correct_avg,
                "std": correct_std,
                "samples": correct_features
            },
            "misplaced": {
                "avg": misplaced_avg,
                "std": misplaced_std,
                "samples": misplaced_features
            }
        },
        "feature_importance": {k: v for k, v in sorted_features},
        "thresholds": {
            k: {
                "value": v["threshold"],
                "direction": v["direction"],
                "score": v["fisher_score"]
            }
            for k, v in sorted_features if v["fisher_score"] > 0.1
        }
    }

    os.makedirs(os.path.dirname(PROFILE_PATH), exist_ok=True)
    with open(PROFILE_PATH, 'w') as f:
        json.dump(profile, f, indent=2)

    print(f"  ✅ Profile saved to: {PROFILE_PATH}")

    # Summary
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"  Training samples: {len(correct_features)} correct + {len(misplaced_features)} misplaced")
    print(f"  Training accuracy: {train_accuracy:.1f}%")
    print(f"  Features extracted: {len(feature_keys)}")
    print(f"  Profile saved: {PROFILE_PATH}")
    print(f"\n  The analyzer will now use this trained profile for")
    print(f"  accurate detection on your specific lab layout! 🎯")


if __name__ == "__main__":
    train()
