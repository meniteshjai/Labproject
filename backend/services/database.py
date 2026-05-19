"""
Database service for Smart Lab Chair Monitoring System.
Uses SQLite via aiosqlite for async database operations.
"""

import aiosqlite
import os
import json
from datetime import datetime
from typing import Optional, List, Dict, Any

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chair_monitor.db")


async def get_db() -> aiosqlite.Connection:
    """Get database connection."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    return db


async def init_db():
    """Initialize database tables."""
    db = await get_db()
    try:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                upload_path TEXT NOT NULL,
                result_path TEXT,
                report_path TEXT,
                total_chairs INTEGER DEFAULT 0,
                total_desks INTEGER DEFAULT 0,
                correct_chairs INTEGER DEFAULT 0,
                misplaced_chairs INTEGER DEFAULT 0,
                accuracy REAL DEFAULT 0.0,
                avg_confidence REAL DEFAULT 0.0,
                status TEXT DEFAULT 'pending',
                lab_room TEXT DEFAULT 'Lab 1',
                uploaded_by TEXT DEFAULT 'Admin',
                created_at TEXT NOT NULL,
                completed_at TEXT,
                details_json TEXT
            )
        """)
        await db.commit()
    finally:
        await db.close()


async def create_analysis(
    analysis_id: str,
    filename: str,
    original_filename: str,
    upload_path: str,
    lab_room: str = "Lab 1",
    uploaded_by: str = "Admin"
) -> Dict[str, Any]:
    """Create a new analysis record."""
    db = await get_db()
    try:
        now = datetime.now().isoformat()
        await db.execute(
            """INSERT INTO analyses 
               (id, filename, original_filename, upload_path, status, lab_room, uploaded_by, created_at)
               VALUES (?, ?, ?, ?, 'pending', ?, ?, ?)""",
            (analysis_id, filename, original_filename, upload_path, lab_room, uploaded_by, now)
        )
        await db.commit()
        return {
            "id": analysis_id,
            "filename": filename,
            "original_filename": original_filename,
            "status": "pending",
            "lab_room": lab_room,
            "uploaded_by": uploaded_by,
            "created_at": now
        }
    finally:
        await db.close()


async def update_analysis_results(
    analysis_id: str,
    result_path: str,
    total_chairs: int,
    total_desks: int,
    correct_chairs: int,
    misplaced_chairs: int,
    accuracy: float,
    avg_confidence: float,
    details: Dict[str, Any]
) -> None:
    """Update analysis with detection results."""
    db = await get_db()
    try:
        now = datetime.now().isoformat()
        await db.execute(
            """UPDATE analyses SET
               result_path = ?,
               total_chairs = ?,
               total_desks = ?,
               correct_chairs = ?,
               misplaced_chairs = ?,
               accuracy = ?,
               avg_confidence = ?,
               status = 'completed',
               completed_at = ?,
               details_json = ?
               WHERE id = ?""",
            (result_path, total_chairs, total_desks, correct_chairs, misplaced_chairs,
             accuracy, avg_confidence, now, json.dumps(details), analysis_id)
        )
        await db.commit()
    finally:
        await db.close()


async def update_analysis_status(analysis_id: str, status: str) -> None:
    """Update analysis status."""
    db = await get_db()
    try:
        await db.execute(
            "UPDATE analyses SET status = ? WHERE id = ?",
            (status, analysis_id)
        )
        await db.commit()
    finally:
        await db.close()


async def update_report_path(analysis_id: str, report_path: str) -> None:
    """Update report path for an analysis."""
    db = await get_db()
    try:
        await db.execute(
            "UPDATE analyses SET report_path = ? WHERE id = ?",
            (report_path, analysis_id)
        )
        await db.commit()
    finally:
        await db.close()


async def get_analysis(analysis_id: str) -> Optional[Dict[str, Any]]:
    """Get a single analysis by ID."""
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,))
        row = await cursor.fetchone()
        if row:
            result = dict(row)
            if result.get("details_json"):
                try:
                    result["details"] = json.loads(result["details_json"])
                except json.JSONDecodeError as e:
                    import logging
                    logging.error(f"Failed to parse JSON for analysis {analysis_id}: {e}")
                    result["details"] = None
            return result
        return None
    finally:
        await db.close()


async def delete_analysis(analysis_id: str) -> None:
    """Delete an analysis record from the database."""
    db = await get_db()
    try:
        await db.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
        await db.commit()
    finally:
        await db.close()


async def delete_all_analyses() -> None:
    """Delete all analysis records from the database."""
    db = await get_db()
    try:
        await db.execute("DELETE FROM analyses")
        await db.commit()
    finally:
        await db.close()


async def get_analyses(
    limit: int = 50,
    offset: int = 0,
    status: Optional[str] = None,
    lab_room: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    search: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get list of analyses with optional filters."""
    db = await get_db()
    try:
        query = "SELECT * FROM analyses WHERE 1=1"
        params = []

        if status:
            query += " AND status = ?"
            params.append(status)
        if lab_room:
            query += " AND lab_room = ?"
            params.append(lab_room)
        if date_from:
            query += " AND created_at >= ?"
            params.append(date_from)
        if date_to:
            query += " AND created_at <= ?"
            params.append(date_to)
        if search:
            query += " AND (original_filename LIKE ? OR lab_room LIKE ? OR uploaded_by LIKE ?)"
            search_param = f"%{search}%"
            params.extend([search_param, search_param, search_param])

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            item = dict(row)
            if item.get("details_json"):
                try:
                    item["details"] = json.loads(item["details_json"])
                except json.JSONDecodeError as e:
                    import logging
                    logging.error(f"Failed to parse JSON for analysis {item.get('id')}: {e}")
                    item["details"] = None
            else:
                item["details"] = None
            results.append(item)
        return results
    finally:
        await db.close()


async def get_dashboard_stats() -> Dict[str, Any]:
    """Get aggregate statistics for the dashboard."""
    db = await get_db()
    try:
        # Total analyses
        cursor = await db.execute("SELECT COUNT(*) as count FROM analyses WHERE status = 'completed'")
        row = await cursor.fetchone()
        total_analyses = row["count"] if row else 0

        # Average accuracy
        cursor = await db.execute("SELECT AVG(accuracy) as avg_acc FROM analyses WHERE status = 'completed'")
        row = await cursor.fetchone()
        avg_accuracy = round(row["avg_acc"], 1) if row and row["avg_acc"] else 0

        # Flagged cases (accuracy < 100)
        cursor = await db.execute(
            "SELECT COUNT(*) as count FROM analyses WHERE status = 'completed' AND misplaced_chairs > 0"
        )
        row = await cursor.fetchone()
        flagged_cases = row["count"] if row else 0

        # Total chairs detected
        cursor = await db.execute(
            "SELECT COALESCE(SUM(total_chairs), 0) as total FROM analyses WHERE status = 'completed'"
        )
        row = await cursor.fetchone()
        total_chairs = row["total"] if row else 0

        # Total misplaced
        cursor = await db.execute(
            "SELECT COALESCE(SUM(misplaced_chairs), 0) as total FROM analyses WHERE status = 'completed'"
        )
        row = await cursor.fetchone()
        total_misplaced = row["total"] if row else 0

        # Daily stats (last 30 days)
        cursor = await db.execute("""
            SELECT DATE(created_at) as date, 
                   COUNT(*) as count,
                   AVG(accuracy) as avg_accuracy,
                   SUM(misplaced_chairs) as misplaced
            FROM analyses 
            WHERE status = 'completed' 
            GROUP BY DATE(created_at) 
            ORDER BY date DESC 
            LIMIT 30
        """)
        daily_rows = await cursor.fetchall()
        daily_stats = [dict(r) for r in daily_rows]

        # Lab room stats
        cursor = await db.execute("""
            SELECT lab_room,
                   COUNT(*) as count,
                   AVG(accuracy) as avg_accuracy,
                   SUM(misplaced_chairs) as total_misplaced
            FROM analyses
            WHERE status = 'completed'
            GROUP BY lab_room
            ORDER BY count DESC
        """)
        lab_rows = await cursor.fetchall()
        lab_stats = [dict(r) for r in lab_rows]

        return {
            "total_analyses": total_analyses,
            "avg_accuracy": avg_accuracy,
            "flagged_cases": flagged_cases,
            "total_chairs_detected": total_chairs,
            "total_misplaced": total_misplaced,
            "perfect_rate": round(((total_analyses - flagged_cases) / max(total_analyses, 1)) * 100, 1),
            "daily_stats": daily_stats,
            "lab_stats": lab_stats
        }
    finally:
        await db.close()
