import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parents[1]
SUBMISSION_DIR = Path(os.getenv("SUBMISSION_STORAGE_DIR", REPO_ROOT / "private_submissions")).resolve()
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
ADMIN_TOKEN = os.getenv("SUBMISSION_ADMIN_TOKEN", "")

PUBLIC_DIRS = {
    "Assignments": REPO_ROOT / "Assignments",
    "Slides": REPO_ROOT / "Slides",
    "Notebooks": REPO_ROOT / "Notebooks",
    "Simulation Demos": REPO_ROOT / "Simulation Demos",
}

app = FastAPI(title="MGMT298D Course Site")


def safe_slug(value: str, fallback: str = "submission") -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-._")
    return slug[:80] or fallback


def require_admin_token(request: Request) -> None:
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=404, detail="Not found")

    auth_header = request.headers.get("authorization", "")
    bearer_token = auth_header.removeprefix("Bearer ").strip()
    query_token = request.query_params.get("token", "")

    if ADMIN_TOKEN not in {bearer_token, query_token}:
        raise HTTPException(status_code=401, detail="Unauthorized")


async def read_pdf(file: UploadFile) -> bytes:
    if file.content_type not in {"application/pdf", "application/octet-stream", ""}:
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The uploaded PDF is too large.")
    if not contents.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="Please upload a valid PDF file.")

    return contents


@app.on_event("startup")
def ensure_storage_dir() -> None:
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
def home() -> FileResponse:
    return FileResponse(REPO_ROOT / "index.html")


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/submit")
async def submit_assignment(
    assignment: str = Form(...),
    student_name: str = Form(...),
    student_email: str = Form(...),
    submission_pdf: UploadFile = File(...),
) -> JSONResponse:
    pdf_bytes = await read_pdf(submission_pdf)
    submitted_at = datetime.now(timezone.utc).isoformat()
    submission_id = uuid4().hex

    assignment_slug = safe_slug(assignment, "assignment")
    student_slug = safe_slug(student_name, "student")
    filename = f"{assignment_slug}_{student_slug}_{submission_id}.pdf"
    submission_path = SUBMISSION_DIR / assignment_slug / filename
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_path.write_bytes(pdf_bytes)

    metadata = {
        "id": submission_id,
        "submitted_at": submitted_at,
        "assignment": assignment,
        "student_name": student_name,
        "student_email": student_email,
        "original_filename": submission_pdf.filename,
        "stored_filename": filename,
        "size_bytes": len(pdf_bytes),
    }

    manifest_path = SUBMISSION_DIR / "manifest.jsonl"
    with manifest_path.open("a", encoding="utf-8") as manifest:
        manifest.write(json.dumps(metadata, ensure_ascii=True) + "\n")

    return JSONResponse({"ok": True, "submission_id": submission_id})


@app.get("/admin/submissions")
def list_submissions(request: Request) -> dict:
    require_admin_token(request)
    manifest_path = SUBMISSION_DIR / "manifest.jsonl"
    if not manifest_path.exists():
        return {"submissions": []}

    submissions = []
    with manifest_path.open("r", encoding="utf-8") as manifest:
        for line in manifest:
            line = line.strip()
            if line:
                submissions.append(json.loads(line))

    return {"submissions": submissions}


@app.get("/admin/submissions/{submission_id}/download")
def download_submission(submission_id: str, request: Request) -> FileResponse:
    require_admin_token(request)
    submission_id = safe_slug(submission_id, "")
    if not submission_id:
        raise HTTPException(status_code=404, detail="Not found")

    matches = list(SUBMISSION_DIR.glob(f"*/*_{submission_id}.pdf"))
    if not matches:
        raise HTTPException(status_code=404, detail="Not found")

    return FileResponse(matches[0], media_type="application/pdf", filename=matches[0].name)


for route, directory in PUBLIC_DIRS.items():
    if directory.exists():
        app.mount(f"/{route}", StaticFiles(directory=directory), name=safe_slug(route))
