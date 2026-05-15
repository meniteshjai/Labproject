"""
Smart Lab Chair Monitoring System — FastAPI Backend
Main application with all API endpoints.
"""

import os
import uuid
import shutil
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from services.database import init_db, create_analysis, update_analysis_results, \
    update_analysis_status, update_report_path, get_analysis, get_analyses, get_dashboard_stats
from services.image_processor import ImageProcessor
from services.report_generator import generate_pdf_report
from models.analyzer import get_analyzer, AnalysisResult

import cv2

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")

# Ensure directories exist
for d in [UPLOAD_DIR, RESULTS_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown lifecycle."""
    # Startup
    print("🚀 Initializing Smart Lab Chair Monitoring System...")
    await init_db()
    print("✅ Database initialized")
    # Pre-load the AI model
    analyzer = get_analyzer()
    if analyzer.model:
        print("✅ AI Model (YOLOv8) loaded")
    else:
        print("⚠️ AI Model failed to load — detection will not work")
    print("🟢 System ready!")
    yield
    # Shutdown
    print("👋 Shutting down...")


app = FastAPI(
    title="Smart Lab Chair Monitoring System",
    description="AI-powered chair arrangement detection and analysis",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- API ENDPOINTS ----------

@app.post("/api/upload")
async def upload_image(
    file: UploadFile = File(...),
    lab_room: str = Form(default="Lab 1"),
    uploaded_by: str = Form(default="Admin")
):
    """
    Upload a lab/classroom image for analysis.
    Accepts JPG, JPEG, and PNG files.
    Returns an analysis ID to use for subsequent operations.
    """
    # Validate format
    if not ImageProcessor.validate_format(file.filename):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload JPG, JPEG, or PNG images."
        )

    # Read file
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    if len(contents) > 50 * 1024 * 1024:  # 50MB limit
        raise HTTPException(status_code=400, detail="File too large. Maximum 50MB.")

    # Generate unique ID and filename
    analysis_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ('.jpg', '.jpeg', '.png'):
        ext = '.jpg'
    saved_filename = f"{analysis_id}{ext}"
    upload_path = os.path.join(UPLOAD_DIR, saved_filename)

    # Preprocess and save
    try:
        processed = ImageProcessor.preprocess(contents)
        ImageProcessor.save_image(processed, upload_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Create database record
    record = await create_analysis(
        analysis_id=analysis_id,
        filename=saved_filename,
        original_filename=file.filename,
        upload_path=upload_path,
        lab_room=lab_room,
        uploaded_by=uploaded_by
    )

    return {
        "success": True,
        "message": "Image uploaded successfully",
        "data": record
    }


@app.post("/api/analyze/{analysis_id}")
async def analyze_image(analysis_id: str):
    """
    Run AI analysis on a previously uploaded image.
    Detects chairs and desks, evaluates arrangement, and generates annotated output.
    """
    # Get the analysis record
    record = await get_analysis(analysis_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found")

    if record["status"] == "completed":
        return {
            "success": True,
            "message": "Analysis already completed",
            "data": _format_analysis_response(record)
        }

    # Update status to processing
    await update_analysis_status(analysis_id, "processing")

    try:
        # Load image
        upload_path = record["upload_path"]
        if not os.path.exists(upload_path):
            raise HTTPException(status_code=404, detail="Uploaded image file not found")

        image = cv2.imread(upload_path)
        if image is None:
            raise HTTPException(status_code=400, detail="Could not read uploaded image")

        # Run AI analysis
        analyzer = get_analyzer()
        result: AnalysisResult = analyzer.analyze_arrangement(image)

        # Generate annotated image
        annotated = analyzer.annotate_image(image, result)
        result_filename = f"result_{analysis_id}.jpg"
        result_path = os.path.join(RESULTS_DIR, result_filename)
        ImageProcessor.save_image(annotated, result_path)

        # Generate heatmap (if there are misplaced chairs)
        if result.misplaced_chairs > 0:
            heatmap = analyzer.generate_heatmap(image, result)
            heatmap_path = os.path.join(RESULTS_DIR, f"heatmap_{analysis_id}.jpg")
            ImageProcessor.save_image(heatmap, heatmap_path)

        # Prepare details for storage
        details = {
            "chairs": result.chairs,
            "desks": result.desks,
            "image_width": result.image_width,
            "image_height": result.image_height
        }

        # Update database
        await update_analysis_results(
            analysis_id=analysis_id,
            result_path=result_path,
            total_chairs=result.total_chairs,
            total_desks=result.total_desks,
            correct_chairs=result.correct_chairs,
            misplaced_chairs=result.misplaced_chairs,
            accuracy=result.accuracy,
            avg_confidence=result.avg_confidence,
            details=details
        )

        # Fetch updated record
        updated_record = await get_analysis(analysis_id)

        return {
            "success": True,
            "message": "Analysis completed successfully",
            "data": _format_analysis_response(updated_record)
        }

    except HTTPException:
        raise
    except Exception as e:
        await update_analysis_status(analysis_id, "failed")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/api/results/{analysis_id}")
async def get_results(analysis_id: str):
    """Get analysis results by ID."""
    record = await get_analysis(analysis_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return {
        "success": True,
        "data": _format_analysis_response(record)
    }


@app.get("/api/reports/{analysis_id}/pdf")
async def download_pdf_report(analysis_id: str):
    """Generate and download a PDF report for an analysis."""
    record = await get_analysis(analysis_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found")

    if record["status"] != "completed":
        raise HTTPException(status_code=400, detail="Analysis not completed yet")

    # Check if report already exists
    report_path = record.get("report_path")
    if report_path and os.path.exists(report_path):
        return FileResponse(
            report_path,
            media_type="application/pdf",
            filename=f"chair_analysis_{analysis_id[:8]}.pdf"
        )

    # Generate new report
    report_filename = f"report_{analysis_id}.pdf"
    report_path = os.path.join(REPORTS_DIR, report_filename)

    try:
        generate_pdf_report(
            analysis=record,
            output_path=report_path,
            upload_image_path=record.get("upload_path"),
            result_image_path=record.get("result_path")
        )

        # Save report path in DB
        await update_report_path(analysis_id, report_path)

        return FileResponse(
            report_path,
            media_type="application/pdf",
            filename=f"chair_analysis_{analysis_id[:8]}.pdf"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@app.get("/api/history")
async def get_history(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    status: Optional[str] = Query(default=None),
    lab_room: Optional[str] = Query(default=None),
    date_from: Optional[str] = Query(default=None),
    date_to: Optional[str] = Query(default=None),
    search: Optional[str] = Query(default=None)
):
    """Get analysis history with optional filters."""
    analyses = await get_analyses(
        limit=limit, offset=offset, status=status,
        lab_room=lab_room, date_from=date_from, date_to=date_to,
        search=search
    )

    return {
        "success": True,
        "data": [_format_analysis_response(a) for a in analyses],
        "count": len(analyses)
    }


@app.get("/api/dashboard/stats")
async def dashboard_stats():
    """Get aggregate dashboard statistics."""
    stats = await get_dashboard_stats()
    return {
        "success": True,
        "data": stats
    }


@app.get("/api/images/uploads/{filename}")
async def serve_upload(filename: str):
    """Serve an uploaded image."""
    path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)


@app.get("/api/images/results/{filename}")
async def serve_result(filename: str):
    """Serve an annotated result image."""
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)


# ---------- HELPERS ----------

def _format_analysis_response(record: dict) -> dict:
    """Format a database record for API response."""
    analysis_id = record.get("id", "")
    filename = record.get("filename", "")
    has_result = record.get("result_path") and os.path.exists(record.get("result_path", ""))
    has_heatmap = os.path.exists(os.path.join(RESULTS_DIR, f"heatmap_{analysis_id}.jpg"))

    return {
        "id": analysis_id,
        "original_filename": record.get("original_filename"),
        "status": record.get("status"),
        "lab_room": record.get("lab_room"),
        "uploaded_by": record.get("uploaded_by"),
        "created_at": record.get("created_at"),
        "completed_at": record.get("completed_at"),
        "total_chairs": record.get("total_chairs", 0),
        "total_desks": record.get("total_desks", 0),
        "correct_chairs": record.get("correct_chairs", 0),
        "misplaced_chairs": record.get("misplaced_chairs", 0),
        "accuracy": record.get("accuracy", 0),
        "avg_confidence": record.get("avg_confidence", 0),
        "upload_image_url": f"/api/images/uploads/{filename}" if filename else None,
        "result_image_url": f"/api/images/results/result_{analysis_id}.jpg" if has_result else None,
        "heatmap_image_url": f"/api/images/results/heatmap_{analysis_id}.jpg" if has_heatmap else None,
        "pdf_report_url": f"/api/reports/{analysis_id}/pdf" if record.get("status") == "completed" else None,
        "details": record.get("details"),
    }


# ---------- STATIC FILES (Frontend) ----------

# Mount frontend static files — must be last
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


# ---------- MAIN ----------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
