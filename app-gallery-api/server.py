#!/usr/bin/env python3
"""Small Railway-hosted repository for SSAI app gallery submissions."""

from __future__ import annotations

import html
import json
import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


SUMMARY_PROMPTS = [
    ("problem", "What is the problem?"),
    ("build", "What did you build and how does it work?"),
    ("endUsers", "Who are the end users, and how does your app deliver value to them?"),
    ("limitations", "What are the main limitations of your approach?"),
    ("improvements", "How would you improve your product?"),
]

MAX_BODY_BYTES = 64 * 1024
MAX_SUMMARY_WORDS = 500


def database_path() -> Path:
    configured_path = os.environ.get("DATABASE_PATH")
    if configured_path:
        return Path(configured_path)

    railway_volume_path = Path("/data")
    if railway_volume_path.exists():
        return railway_volume_path / "app_submissions.sqlite3"

    return Path(__file__).resolve().parent / "data" / "app_submissions.sqlite3"


DB_PATH = database_path()


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize(value: Any) -> str:
    return str(value or "").strip()


def word_count(value: str) -> int:
    return len([word for word in re.split(r"\s+", normalize(value)) if word])


def is_public_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def is_google_drive_url(value: str) -> bool:
    if not is_public_url(value):
        return False
    parsed = urlparse(value)
    return parsed.netloc.endswith("drive.google.com") and bool(re.search(r"[-\w]{25,}", value))


def ensure_database() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS submissions (
                id TEXT PRIMARY KEY,
                app_name TEXT NOT NULL,
                team_members TEXT NOT NULL,
                live_url TEXT NOT NULL,
                video_url TEXT NOT NULL,
                keywords_json TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                writeup TEXT NOT NULL,
                submitted_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_submissions_submitted_at ON submissions(submitted_at)"
        )


def connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def row_to_submission(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "appName": row["app_name"],
        "teamMembers": row["team_members"],
        "liveUrl": row["live_url"],
        "videoUrl": row["video_url"],
        "keywords": json.loads(row["keywords_json"]),
        "summary": json.loads(row["summary_json"]),
        "writeup": row["writeup"],
        "submittedAt": row["submitted_at"],
    }


def list_submissions() -> list[dict[str, Any]]:
    ensure_database()
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT id, app_name, team_members, live_url, video_url, keywords_json,
                   summary_json, writeup, submitted_at
            FROM submissions
            ORDER BY submitted_at DESC
            """
        ).fetchall()
    return [row_to_submission(row) for row in rows]


def combine_summary(summary: dict[str, str]) -> str:
    sections = [
        f"{label}\n{summary[key]}"
        for key, label in SUMMARY_PROMPTS
        if normalize(summary.get(key))
    ]
    return "\n\n".join(sections)


def validate_submission(payload: dict[str, Any]) -> dict[str, Any]:
    app_name = normalize(payload.get("appName"))
    team_members = normalize(payload.get("teamMembers"))
    live_url = normalize(payload.get("liveUrl"))
    video_url = normalize(payload.get("videoUrl"))
    summary_source = payload.get("summary")

    if not isinstance(summary_source, dict):
        summary_source = {}

    if isinstance(payload.get("keywords"), list):
        keywords = [normalize(keyword) for keyword in payload.get("keywords")]
    else:
        keywords = [normalize(keyword) for keyword in normalize(payload.get("keywords")).split(",")]
    keywords = [keyword for keyword in keywords if keyword]

    summary = {
        key: normalize(summary_source.get(key) or payload.get(key))
        for key, _label in SUMMARY_PROMPTS
    }
    writeup = combine_summary(summary)

    errors: list[str] = []
    if not app_name:
        errors.append("App name is required.")
    if not team_members:
        errors.append("Team members are required.")
    if not is_public_url(live_url):
        errors.append("Live app URL must be a valid http or https URL.")
    if not is_google_drive_url(video_url):
        errors.append("Pitch video must be a valid Google Drive URL.")
    if len(keywords) < 3 or len(keywords) > 5:
        errors.append("Please include 3 to 5 keywords.")
    if any(not summary[key] for key, _label in SUMMARY_PROMPTS):
        errors.append("All five summary questions are required.")
    if word_count(writeup) > MAX_SUMMARY_WORDS:
        errors.append("Written summary must be 500 words or fewer in total.")

    field_lengths = {
        "App name": (app_name, 140),
        "Team members": (team_members, 240),
        "Live app URL": (live_url, 600),
        "Pitch video URL": (video_url, 600),
        "Written summary": (writeup, 8000),
    }
    for label, (value, limit) in field_lengths.items():
        if len(value) > limit:
            errors.append(f"{label} is too long.")
    if any(len(keyword) > 48 for keyword in keywords):
        errors.append("Keywords must be 48 characters or fewer.")

    if errors:
        raise ValueError(" ".join(errors))

    submitted_at = now_iso()
    return {
        "id": str(uuid.uuid4()),
        "appName": app_name,
        "teamMembers": team_members,
        "liveUrl": live_url,
        "videoUrl": video_url,
        "keywords": keywords,
        "summary": summary,
        "writeup": writeup,
        "submittedAt": submitted_at,
    }


def save_submission(submission: dict[str, Any]) -> dict[str, Any]:
    ensure_database()
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO submissions (
                id, app_name, team_members, live_url, video_url, keywords_json,
                summary_json, writeup, submitted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                submission["id"],
                submission["appName"],
                submission["teamMembers"],
                submission["liveUrl"],
                submission["videoUrl"],
                json.dumps(submission["keywords"], ensure_ascii=True),
                json.dumps(submission["summary"], ensure_ascii=True),
                submission["writeup"],
                submission["submittedAt"],
            ),
        )
    return submission


def render_repository_page(submissions: list[dict[str, Any]]) -> bytes:
    rows = []
    for app in submissions:
        keywords = ", ".join(app["keywords"])
        rows.append(
            "<tr>"
            f"<td>{html.escape(app['appName'])}</td>"
            f"<td>{html.escape(app['teamMembers'])}</td>"
            f"<td><a href=\"{html.escape(app['liveUrl'])}\">Open app</a></td>"
            f"<td><a href=\"{html.escape(app['videoUrl'])}\">Pitch video</a></td>"
            f"<td>{html.escape(keywords)}</td>"
            f"<td>{html.escape(app['submittedAt'])}</td>"
            "</tr>"
        )

    table_rows = "\n".join(rows) or "<tr><td colspan=\"6\">No submissions yet.</td></tr>"
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SSAI App Submission Repository</title>
  <style>
    body {{ margin: 0; padding: 32px; color: #313339; font-family: Arial, sans-serif; line-height: 1.5; }}
    main {{ max-width: 1120px; margin: 0 auto; }}
    h1 {{ margin: 0 0 8px; color: #003b5c; }}
    p {{ margin: 0 0 18px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid #d9dee5; padding: 10px; text-align: left; vertical-align: top; }}
    th {{ color: #003b5c; font-size: 0.86rem; text-transform: uppercase; }}
    a {{ color: #003b5c; font-weight: 700; }}
  </style>
</head>
<body>
  <main>
    <h1>SSAI App Submission Repository</h1>
    <p>{len(submissions)} submissions. Raw JSON is available at <a href="/submissions">/submissions</a>.</p>
    <table>
      <thead>
        <tr>
          <th>App</th>
          <th>Team</th>
          <th>App URL</th>
          <th>Video</th>
          <th>Keywords</th>
          <th>Submitted</th>
        </tr>
      </thead>
      <tbody>{table_rows}</tbody>
    </table>
  </main>
</body>
</html>"""
    return page.encode("utf-8")


class AppGalleryHandler(BaseHTTPRequestHandler):
    server_version = "SSAIAppGallery/1.0"

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", os.environ.get("ALLOWED_ORIGIN", "*"))
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self.write_json(HTTPStatus.OK, {"ok": True, "database": str(DB_PATH)})
            return

        if parsed.path == "/submissions":
            self.write_json(HTTPStatus.OK, {"submissions": list_submissions()})
            return

        if parsed.path == "/":
            body = render_repository_page(list_submissions())
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.write_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/submissions":
            self.write_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0

        if content_length <= 0 or content_length > MAX_BODY_BYTES:
            self.write_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid request body."})
            return

        try:
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Request body must be a JSON object.")
            submission = save_submission(validate_submission(payload))
        except json.JSONDecodeError:
            self.write_json(HTTPStatus.BAD_REQUEST, {"error": "Request body must be valid JSON."})
            return
        except ValueError as error:
            self.write_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        except sqlite3.Error:
            self.write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Submission repository is unavailable."})
            return

        self.write_json(HTTPStatus.CREATED, {"submission": submission})


def main() -> None:
    ensure_database()
    port = int(os.environ.get("PORT", "8787"))
    server = ThreadingHTTPServer(("0.0.0.0", port), AppGalleryHandler)
    print(f"SSAI app gallery repository listening on port {port}")
    print(f"Database: {DB_PATH}")
    server.serve_forever()


if __name__ == "__main__":
    main()
