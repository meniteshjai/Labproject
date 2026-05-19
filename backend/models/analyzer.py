"""
AI Chair Detection & Alignment Analysis Engine.
Uses Groq (Llama 4 Scout Vision) to analyze chair arrangement in lab environments.
Sends the photo to Groq and displays the AI's analysis result directly.
"""

import cv2
import numpy as np
import base64
import json
import logging
import os
from time import perf_counter
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field, asdict
from types import SimpleNamespace

logger = logging.getLogger("backend.analyzer")

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    Groq = None
    GROQ_AVAILABLE = False

try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except ImportError:
    InferenceClient = None
    HF_AVAILABLE = False


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
    alignment_score: float
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
    ai_description: str = ""
    ai_provider: str = "unknown"
    ai_model: str = ""


class ChairAnalyzer:
    """
    AI-powered chair arrangement analyzer using Groq Vision API.
    Uses meta-llama/llama-4-scout-17b-16e-instruct for fast, accurate vision.
    """

    def __init__(self):
        self.provider = os.environ.get("AI_PROVIDER", "auto").strip().lower()
        self.groq_api_key = os.environ.get("GROQ_API_KEY")
        self.hf_token = os.environ.get("HF_TOKEN")
        self.hf_model_name = os.environ.get("HF_MODEL", "zai-org/GLM-OCR")
        self.groq_model_name = "meta-llama/llama-4-scout-17b-16e-instruct"
        self.model_name = self.hf_model_name if self.provider in ("huggingface", "auto") else self.groq_model_name
        self.client = None
        self.hf_client = None
        self.is_ready_flag = False

        if self.provider in ("huggingface", "auto"):
            if not self.hf_token:
                logger.warning("⚠️ HF_TOKEN not set. Hugging Face API calls will fail.")
            elif not HF_AVAILABLE:
                logger.warning("⚠️ huggingface_hub package not installed.")
            else:
                self.is_ready_flag = True
                logger.info("✅ Hugging Face Inference configured (%s)", self.hf_model_name)

        if self.provider in ("groq", "auto"):
            if not self.groq_api_key:
                logger.warning("⚠️ GROQ_API_KEY not set. Groq API calls will fail.")
            elif not GROQ_AVAILABLE:
                logger.warning("⚠️ groq package not installed.")
            else:
                self.client = Groq(api_key=self.groq_api_key)
                self.is_ready_flag = True
                logger.info("✅ Groq API Client initialized (%s)", self.groq_model_name)

        # YOLO model (lazy-loaded)
        self._yolo_model = None
        try:
            from ultralytics import YOLO  # type: ignore
            self._YOLO_CLASS = YOLO
        except Exception:
            self._YOLO_CLASS = None

    @property
    def is_ready(self) -> bool:
        return self.is_ready_flag

    def detect_objects(self, image: np.ndarray) -> List[Any]:
        """Lightweight fallback object detector stub.

        Returns an empty list when no detector is configured. This keeps
        the analyzer usable in environments without Groq or a local detector.
        """
        # Prefer ultralytics YOLO if available and a model file exists.
        if self._YOLO_CLASS is None:
            return []

        # Find model path candidates relative to this file and workspace root
        candidates = [
            os.path.join(os.path.dirname(__file__), '..', 'yolov8n.pt'),
            os.path.join(os.path.dirname(__file__), '..', '..', 'yolov8n.pt'),
            os.path.join(os.getcwd(), 'backend', 'yolov8n.pt'),
            os.path.join(os.getcwd(), 'yolov8n.pt'),
        ]
        model_path = None
        for p in candidates:
            p = os.path.normpath(p)
            if os.path.exists(p):
                model_path = p
                break

        if model_path is None:
            return []

        # Lazy load model
        if self._yolo_model is None:
            try:
                self._yolo_model = self._YOLO_CLASS(model_path)
            except Exception:
                return []

        # Run inference
        try:
            results = self._yolo_model.predict(source=image, verbose=False)
        except TypeError:
            # older API fallback
            results = self._yolo_model(image)

        detections: List[Any] = []
        # results may be a list of Result objects (one per image)
        for res in results:
            boxes = getattr(res, 'boxes', None)
            if boxes is None:
                continue
            xyxy = getattr(boxes, 'xyxy', None)
            cls = getattr(boxes, 'cls', None)
            conf = getattr(boxes, 'conf', None)
            # Some ultralytics versions use numpy arrays
            if xyxy is None:
                continue

            try:
                arr_xyxy = xyxy.cpu().numpy() if hasattr(xyxy, 'cpu') else (xyxy.numpy() if hasattr(xyxy, 'numpy') else xyxy)
            except Exception:
                arr_xyxy = xyxy

            try:
                arr_cls = cls.cpu().numpy() if hasattr(cls, 'cpu') else (cls.numpy() if hasattr(cls, 'numpy') else cls)
            except Exception:
                arr_cls = cls

            try:
                arr_conf = conf.cpu().numpy() if hasattr(conf, 'cpu') else (conf.numpy() if hasattr(conf, 'numpy') else conf)
            except Exception:
                arr_conf = conf

            for i, box in enumerate(arr_xyxy):
                x1, y1, x2, y2 = map(int, box[:4])
                width = max(1, x2 - x1)
                height = max(1, y2 - y1)
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                class_id = int(arr_cls[i]) if arr_cls is not None else -1
                confidence = float(arr_conf[i]) if arr_conf is not None else 0.0

                det = SimpleNamespace()
                det.class_id = class_id
                det.bbox = (x1, y1, x2, y2)
                det.center = center
                det.width = width
                det.height = height
                det.confidence = confidence
                detections.append(det)

        return detections

    def _estimate_tilt_angle(self, obj: Any) -> float:
        """Estimate chair tilt angle. Stub returns 0.0 for compatibility."""
        return 0.0

    def _is_chair_pulled_out_of_row(self, obj: Any, chairs: List[Any]) -> bool:
        """Determine if a chair is pulled out. Stub returns False for compatibility."""
        return False

    def _call_groq(self, image: np.ndarray) -> dict:
        """Sends image to Groq Vision API and returns structured JSON."""
        if not self.client:
            raise RuntimeError("Groq Client not initialized. Check GROQ_API_KEY.")

        h, w = image.shape[:2]

        # Encode image to base64 JPEG
        success, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            raise ValueError("Could not encode image.")
        b64_image = base64.b64encode(encoded.tobytes()).decode('utf-8')

        prompt = (
            "You are a lab chair monitoring AI. Look at this photo of a computer lab.\n\n"
            "The lab has computer desks arranged in rows. Each row has 7 chairs.\n"
            "There may be 1 row (7 chairs) or 2 rows (14 chairs) visible.\n\n"
            "Analyze the image and tell me:\n"
            "1. How many total chairs are visible?\n"
            "2. How many are MISPLACED (pulled out from desk, rotated, in the aisle)?\n"
            "3. For each misplaced chair, describe its position and what's wrong.\n\n"
            "A chair is PROPERLY POSITIONED if it is tucked under the desk close to the partition.\n"
            "A chair is MISPLACED if it is pulled out into the aisle, rotated sideways, or far from the desk.\n\n"
            "Be thorough — check EVERY chair carefully.\n\n"
            "Return ONLY valid JSON matching this exact schema:\n"
            "{\n"
            '  "total_chairs": 14,\n'
            '  "rows_visible": 2,\n'
            '  "misplaced_chairs": [\n'
            '    {"chair_number": 3, "position": "left row, 3rd from front", "issue": "pulled out into aisle"},\n'
            '    {"chair_number": 5, "position": "right row, 2nd from front", "issue": "rotated sideways"}\n'
            '  ],\n'
            '  "properly_placed_count": 12,\n'
            '  "summary": "2 out of 14 chairs are misplaced. Chair 3 is pulled into the aisle. Chair 5 is rotated."\n'
            "}"
        )

        start_time = perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.groq_model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_image}",
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
                temperature=0.0,
                max_tokens=2048,
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content
            duration_ms = (perf_counter() - start_time) * 1000.0
            try:
                groq_response = json.loads(raw)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Groq JSON response: {e}. Raw response: {raw[:500]}")
                raise RuntimeError(f"Groq API returned invalid JSON: {str(e)}")

            logger.info(
                "Groq API response: model=%s duration_ms=%.1f total_chairs=%s misplaced=%s proper_count=%s",
                self.model_name,
                duration_ms,
                groq_response.get("total_chairs"),
                len(groq_response.get("misplaced_chairs", [])) if isinstance(groq_response.get("misplaced_chairs"), list) else groq_response.get("misplaced_chairs"),
                groq_response.get("properly_placed_count"),
            )
            logger.debug("Groq raw response: %s", raw)
            return groq_response

        except Exception as e:
            logger.exception("Groq API Error")
            error_msg = str(e).lower()
            if "invalid api key" in error_msg or "authentication" in error_msg:
                raise RuntimeError("INVALID_API_KEY")
            if "rate limit" in error_msg or "quota" in error_msg or "429" in error_msg:
                raise RuntimeError("QUOTA_EXHAUSTED")
            raise RuntimeError(f"Groq API failed: {str(e)}")

    def _call_hf(self, image: np.ndarray) -> str:
        """Sends image to Hugging Face Inference API and returns raw text."""
        if not self.hf_token:
            raise RuntimeError("HF_TOKEN_MISSING")
        if not HF_AVAILABLE or InferenceClient is None:
            raise RuntimeError("HUGGINGFACE_HUB_NOT_INSTALLED")

        if self.hf_client is None:
            self.hf_client = InferenceClient(token=self.hf_token)

        success, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            raise ValueError("Could not encode image for Hugging Face Inference.")
        image_bytes = encoded.tobytes()

        start_time = perf_counter()
        try:
            raw_text = self.hf_client.image_to_text(image_bytes, model=self.hf_model_name)
        except Exception as e:
            logger.exception("Hugging Face Inference error")
            error_msg = str(e).lower()
            if "invalid" in error_msg or "authentication" in error_msg:
                raise RuntimeError("INVALID_API_KEY")
            if "rate limit" in error_msg or "quota" in error_msg or "429" in error_msg:
                raise RuntimeError("QUOTA_EXHAUSTED")
            raise RuntimeError(f"Hugging Face Inference failed: {str(e)}")

        duration_ms = (perf_counter() - start_time) * 1000.0
        logger.info(
            "Hugging Face response: model=%s duration_ms=%.1f output_len=%d",
            self.hf_model_name,
            duration_ms,
            len(raw_text) if raw_text is not None else 0,
        )
        logger.debug("Hugging Face raw text: %s", raw_text)
        return raw_text if raw_text is not None else ""

    def analyze_arrangement(self, image: np.ndarray) -> AnalysisResult:
        """Full analysis pipeline using Groq or Hugging Face Vision API."""
        h, w = image.shape[:2]

        used_provider = self.provider
        try:
            if self.provider in ("huggingface", "auto"):
                try:
                    raw_text = self._call_hf(image)
                    detected_chairs = [o for o in self.detect_objects(image) if getattr(o, "class_id", None) == 56]
                    total_chairs = len(detected_chairs)
                    groq_response = {
                        "total_chairs": total_chairs,
                        "rows_visible": 0,
                        "misplaced_chairs": [],
                        "properly_placed_count": total_chairs,
                        "summary": raw_text,
                    }
                    scene_class = "huggingface_image_to_text"
                    scene_confidence = 50.0
                    used_provider = "huggingface"
                except RuntimeError as hf_err:
                    if self.provider == "huggingface":
                        raise
                    logger.warning("Hugging Face failed, falling back to Groq: %s", hf_err)
                    groq_response = self._call_groq(image)
                    scene_class = "groq_analyzed"
                    scene_confidence = 90.0
                    used_provider = "groq"
            else:
                groq_response = self._call_groq(image)
                scene_class = "groq_analyzed"
                scene_confidence = 90.0
                used_provider = "groq"
        except RuntimeError as e:
            err = str(e)
            if err == "INVALID_API_KEY":
                scene_class = "invalid_api_key"
            elif err == "QUOTA_EXHAUSTED":
                scene_class = "quota_exhausted"
            else:
                scene_class = "error"
            logger.warning("Analysis error: %s", e)
            return AnalysisResult(
                total_chairs=0, total_desks=0, correct_chairs=0, misplaced_chairs=0,
                accuracy=0.0, avg_confidence=0.0, chairs=[], desks=[],
                image_width=w, image_height=h, scene_classification=scene_class,
                ai_provider=used_provider,
                ai_model=self.hf_model_name if used_provider == "huggingface" else self.groq_model_name
            )

        # Parse response
        total_chairs = groq_response.get("total_chairs", 0)
        misplaced_list = groq_response.get("misplaced_chairs", [])
        # Handle case where misplaced_chairs might be an int instead of list
        if isinstance(misplaced_list, int):
            misplaced_count = misplaced_list
            misplaced_list = []
        else:
            misplaced_count = len(misplaced_list)
        
        properly_placed = groq_response.get("properly_placed_count", total_chairs - misplaced_count)
        summary = groq_response.get("summary", "")
        rows_visible = groq_response.get("rows_visible", 1)

        logger.info(
            "AnalysisResult details: provider=%s model=%s total_chairs=%d misplaced=%d accuracy=%.1f scene=%s summary_snippet=%s",
            used_provider,
            self.model_name,
            total_chairs,
            misplaced_count,
            (properly_placed / max(total_chairs, 1)) * 100.0,
            scene_class,
            repr(summary[:180].replace("\n", " "))
        )

        # Build chair analysis list for compatibility with existing frontend
        chair_analyses = []
        
        # Add properly placed chairs
        misplaced_numbers = set()
        for mc in misplaced_list:
            if isinstance(mc, dict):
                misplaced_numbers.add(mc.get("chair_number", 0))
        
        for i in range(1, total_chairs + 1):
            is_misplaced = i in misplaced_numbers
            issues = []
            if is_misplaced:
                # Find the matching misplaced entry
                for mc in misplaced_list:
                    if isinstance(mc, dict) and mc.get("chair_number") == i:
                        issue = mc.get("issue", "misplaced")
                        position = mc.get("position", "")
                        if position:
                            issues.append(f"{issue} ({position})")
                        else:
                            issues.append(issue)
                        break
            
            chair_analyses.append(ChairAnalysis(
                chair_id=i,
                bbox=(0, 0, 0, 0),
                center=(0, 0),
                confidence=0.9,
                is_properly_arranged=not is_misplaced,
                nearest_desk_id=None,
                distance_to_desk=None,
                alignment_score=35.0 if is_misplaced else 95.0,
                issues=issues
            ))

        accuracy = (properly_placed / max(total_chairs, 1)) * 100.0
        logger.info(
            "AnalysisResult generated: provider=%s model=%s total_chairs=%d misplaced=%d accuracy=%.1f scene=%s",
            self.provider,
            self.model_name,
            total_chairs,
            misplaced_count,
            round(accuracy, 1),
            scene_class
        )

        return AnalysisResult(
            total_chairs=total_chairs,
            total_desks=0,
            correct_chairs=properly_placed,
            misplaced_chairs=misplaced_count,
            accuracy=round(accuracy, 1),
            avg_confidence=90.0,
            chairs=[asdict(c) for c in chair_analyses],
            desks=[],
            image_width=w,
            image_height=h,
            scene_classification=scene_class,
            scene_confidence=scene_confidence,
            ai_description=summary,
            ai_provider=used_provider,
            ai_model=self.hf_model_name if used_provider == "huggingface" else self.groq_model_name
        )

    def annotate_image(self, image: np.ndarray, result: AnalysisResult) -> np.ndarray:
        """
        Draw a clean summary overlay on the original image.
        No bounding boxes — just the stats and AI description.
        """
        annotated = image.copy()
        h, w = annotated.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Calculate overlay height based on content
        lines = []
        lines.append(f"Total Chairs: {result.total_chairs}")
        lines.append(f"Properly Arranged: {result.correct_chairs}  |  Misplaced: {result.misplaced_chairs}")
        
        if result.accuracy >= 80:
            acc_text = f"Arrangement Accuracy: {result.accuracy}%"
        else:
            acc_text = f"Arrangement Accuracy: {result.accuracy}%"
        lines.append(acc_text)

        # Add misplaced chair details
        misplaced_details = []
        for chair in result.chairs:
            if not chair["is_properly_arranged"] and chair["issues"]:
                misplaced_details.append(f"  Chair #{chair['chair_id']}: {', '.join(chair['issues'])}")

        overlay_height = 100 + len(misplaced_details) * 22 + 25
        overlay_width = min(w, 600)

        # Draw semi-transparent overlay
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (overlay_width, overlay_height), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.8, annotated, 0.2, 0, annotated)

        # Draw summary text
        y_pos = 24
        cv2.putText(annotated, lines[0], (10, y_pos), font, 0.62, (255, 255, 255), 1)
        y_pos += 24
        cv2.putText(annotated, lines[1], (10, y_pos), font, 0.55, (255, 255, 255), 1)
        y_pos += 28

        acc_color = (0, 220, 80) if result.accuracy >= 80 else (0, 180, 255) if result.accuracy >= 50 else (0, 60, 230)
        cv2.putText(annotated, acc_text, (10, y_pos), font, 0.72, acc_color, 2)
        y_pos += 24

        cv2.putText(annotated, "AI: Groq (Llama 4 Scout Vision)", (10, y_pos), font, 0.42, (160, 160, 160), 1)
        y_pos += 20

        # Draw misplaced chair details
        if misplaced_details:
            cv2.putText(annotated, "Misplaced:", (10, y_pos), font, 0.5, (0, 100, 255), 1)
            y_pos += 20
            for detail in misplaced_details:
                cv2.putText(annotated, detail, (10, y_pos), font, 0.42, (0, 130, 255), 1)
                y_pos += 22

        return annotated

    def generate_heatmap(self, image: np.ndarray, result: AnalysisResult) -> np.ndarray:
        """Generate a simple heatmap overlay. Since we don't have bboxes, just return the annotated image."""
        # Without precise bounding boxes, return a copy of the image with status overlay
        return self.annotate_image(image, result)


# Singleton
_analyzer_instance = None


def get_analyzer() -> ChairAnalyzer:
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = ChairAnalyzer()
    return _analyzer_instance


def reload_analyzer():
    global _analyzer_instance
    _analyzer_instance = ChairAnalyzer()
    return _analyzer_instance
