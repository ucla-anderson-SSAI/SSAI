#!/usr/bin/env python3
"""Small Railway-hosted repository for SSAI app gallery submissions."""

from __future__ import annotations

import html
import hmac
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
MAX_SUMMARY_WORDS = 600
VOTE_CHOICE_COUNT = 3
ID_PATTERN = re.compile(r"^[a-f0-9-]{36}$", re.IGNORECASE)
UID_PATTERN = re.compile(r"^\d{1,32}$")


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


def configured_admin_pin() -> str:
    return normalize(os.environ.get("ADMIN_PIN") or os.environ.get("ADMIN_TOKEN"))


def request_token(headers: Any) -> str:
    authorization = normalize(headers.get("Authorization"))
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return normalize(headers.get("X-Admin-Token"))


def admin_pin_is_valid(headers: Any) -> bool:
    pin = configured_admin_pin()
    supplied_pin = request_token(headers)
    return bool(pin and supplied_pin and hmac.compare_digest(pin, supplied_pin))


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
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS votes (
                id TEXT PRIMARY KEY,
                student_uid TEXT NOT NULL UNIQUE,
                first_submission_id TEXT NOT NULL,
                first_app_name TEXT NOT NULL,
                second_submission_id TEXT NOT NULL,
                second_app_name TEXT NOT NULL,
                third_submission_id TEXT NOT NULL,
                third_app_name TEXT NOT NULL,
                submitted_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_votes_submitted_at ON votes(submitted_at)"
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


def summary_word_count(summary: dict[str, str]) -> int:
    return sum(word_count(summary.get(key, "")) for key, _label in SUMMARY_PROMPTS)


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
    if summary_word_count(summary) > MAX_SUMMARY_WORDS:
        errors.append(f"Written summary must be {MAX_SUMMARY_WORDS} words or fewer in total.")

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


def delete_submission(submission_id: str) -> bool:
    ensure_database()
    with connect() as connection:
        result = connection.execute("DELETE FROM submissions WHERE id = ?", (submission_id,))
    return result.rowcount > 0


def submission_names_by_id(submission_ids: list[str]) -> dict[str, str]:
    ensure_database()
    if not submission_ids:
        return {}

    placeholders = ",".join("?" for _id in submission_ids)
    with connect() as connection:
        rows = connection.execute(
            f"SELECT id, app_name FROM submissions WHERE id IN ({placeholders})",
            submission_ids,
        ).fetchall()
    return {row["id"]: row["app_name"] for row in rows}


def validate_vote(payload: dict[str, Any]) -> dict[str, Any]:
    student_uid = normalize(payload.get("studentUid") or payload.get("uid"))
    raw_choices = payload.get("choices")
    if not isinstance(raw_choices, list):
        raw_choices = [payload.get("choice1"), payload.get("choice2"), payload.get("choice3")]

    choices = [normalize(choice) for choice in raw_choices if normalize(choice)]

    errors: list[str] = []
    if not UID_PATTERN.fullmatch(student_uid):
        errors.append("Student UID must contain numbers only.")
    if len(choices) != VOTE_CHOICE_COUNT:
        errors.append("Please choose three submissions.")
    if len(set(choices)) != len(choices):
        errors.append("Please choose three different submissions.")
    if any(not ID_PATTERN.fullmatch(choice) for choice in choices):
        errors.append("One or more selected submissions are invalid.")

    app_names = submission_names_by_id(choices) if not errors else {}
    missing_choices = [choice for choice in choices if choice not in app_names]
    if missing_choices:
        errors.append("One or more selected submissions could not be found.")

    if errors:
        raise ValueError(" ".join(errors))

    return {
        "id": str(uuid.uuid4()),
        "studentUid": student_uid,
        "choices": [
            {"submissionId": choice, "appName": app_names[choice]}
            for choice in choices
        ],
        "submittedAt": now_iso(),
    }


def save_vote(vote: dict[str, Any]) -> dict[str, Any]:
    ensure_database()
    choices = vote["choices"]
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO votes (
                id, student_uid, first_submission_id, first_app_name,
                second_submission_id, second_app_name, third_submission_id,
                third_app_name, submitted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(student_uid) DO UPDATE SET
                id = excluded.id,
                first_submission_id = excluded.first_submission_id,
                first_app_name = excluded.first_app_name,
                second_submission_id = excluded.second_submission_id,
                second_app_name = excluded.second_app_name,
                third_submission_id = excluded.third_submission_id,
                third_app_name = excluded.third_app_name,
                submitted_at = excluded.submitted_at
            """,
            (
                vote["id"],
                vote["studentUid"],
                choices[0]["submissionId"],
                choices[0]["appName"],
                choices[1]["submissionId"],
                choices[1]["appName"],
                choices[2]["submissionId"],
                choices[2]["appName"],
                vote["submittedAt"],
            ),
        )
    return vote


def row_to_vote(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "studentUid": row["student_uid"],
        "choices": [
            {"submissionId": row["first_submission_id"], "appName": row["first_app_name"]},
            {"submissionId": row["second_submission_id"], "appName": row["second_app_name"]},
            {"submissionId": row["third_submission_id"], "appName": row["third_app_name"]},
        ],
        "submittedAt": row["submitted_at"],
    }


def list_votes() -> list[dict[str, Any]]:
    ensure_database()
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT id, student_uid, first_submission_id, first_app_name,
                   second_submission_id, second_app_name, third_submission_id,
                   third_app_name, submitted_at
            FROM votes
            ORDER BY submitted_at DESC
            """
        ).fetchall()
    return [row_to_vote(row) for row in rows]


def vote_results() -> dict[str, Any]:
    votes = list_votes()
    totals_by_submission: dict[str, dict[str, Any]] = {}
    for vote in votes:
        for choice in vote["choices"]:
            submission_id = choice["submissionId"]
            total = totals_by_submission.setdefault(
                submission_id,
                {"submissionId": submission_id, "appName": choice["appName"], "count": 0},
            )
            total["count"] += 1

    totals = sorted(
        totals_by_submission.values(),
        key=lambda item: (-int(item["count"]), str(item["appName"]).lower()),
    )
    return {"votes": votes, "totals": totals}


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


def render_admin_page() -> bytes:
    page = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SSAI App Gallery Admin</title>
  <style>
    :root { --blue: #003b5c; --ucla-blue: #2774ae; --seafoam: #4e7e6b; --slate: #313339; --line: #d9dee5; --surface: #f7f8fa; --danger: #9f2d1d; }
    * { box-sizing: border-box; }
    body { margin: 0; padding: 32px; color: var(--slate); font-family: Arial, sans-serif; line-height: 1.5; }
    main { max-width: 920px; margin: 0 auto; }
    h1 { margin: 0 0 8px; color: var(--blue); }
    h2 { margin: 28px 0 12px; color: var(--blue); font-size: 1.2rem; }
    p { margin: 0 0 18px; }
    table { width: 100%; border-collapse: collapse; margin-top: 8px; }
    th, td { border-bottom: 1px solid var(--line); padding: 9px 8px; text-align: left; vertical-align: top; }
    th { color: var(--blue); font-size: 0.78rem; text-transform: uppercase; }
    label { display: grid; gap: 6px; color: var(--blue); font-weight: 700; }
    input { width: 100%; min-height: 42px; border: 1px solid var(--line); border-radius: 3px; padding: 9px 12px; font: inherit; }
    button { min-height: 38px; border: 1px solid var(--line); border-radius: 3px; padding: 8px 12px; color: var(--blue); background: white; cursor: pointer; font: inherit; font-weight: 700; }
    button.primary { border-color: var(--seafoam); color: white; background: var(--seafoam); }
    button.danger { border-color: var(--danger); color: var(--danger); }
    button:disabled { cursor: not-allowed; opacity: 0.6; }
    .controls { display: grid; gap: 12px; max-width: 520px; margin-top: 20px; }
    .status { min-height: 24px; color: #5c6670; font-size: 0.94rem; }
    .status.error { color: var(--danger); font-weight: 700; }
    .submission { display: grid; gap: 8px; border: 1px solid var(--line); border-radius: 5px; padding: 14px; margin-top: 10px; background: white; }
    .submission strong { color: var(--blue); font-size: 1.05rem; }
    .meta { color: #5c6670; font-weight: 700; }
    .links { display: flex; flex-wrap: wrap; gap: 12px; }
    .vote-results { display: grid; gap: 16px; margin-top: 10px; }
    .ballots { display: grid; gap: 10px; }
    .ballot { border: 1px solid var(--line); border-radius: 5px; padding: 12px; background: var(--surface); }
    .ballot strong { color: var(--blue); }
    .ballot ol { margin: 8px 0 0; padding-left: 20px; }
    a { color: var(--ucla-blue); font-weight: 700; }
  </style>
</head>
<body>
  <main>
    <h1>SSAI App Gallery Admin</h1>
    <p>Enter the 4-digit admin PIN to delete submissions and view voting results.</p>

    <div class="controls">
      <label>
        Admin PIN
        <input id="adminPin" type="password" inputmode="numeric" pattern="[0-9]{4}" maxlength="4" autocomplete="current-password">
      </label>
      <button class="primary" id="loadButton" type="button">Load admin data</button>
      <div class="status" id="status"></div>
    </div>

    <h2>Vote Results</h2>
    <div class="vote-results" id="voteResults"></div>

    <h2>Submissions</h2>
    <div id="submissions"></div>
  </main>

  <script>
    const pinInput = document.querySelector("#adminPin");
    const loadButton = document.querySelector("#loadButton");
    const statusBox = document.querySelector("#status");
    const submissionsBox = document.querySelector("#submissions");
    const voteResultsBox = document.querySelector("#voteResults");

    function setStatus(message, isError = false) {
      statusBox.textContent = message;
      statusBox.className = isError ? "status error" : "status";
    }

    function text(value) {
      return String(value || "").trim();
    }

    async function jsonOrError(response, fallbackMessage) {
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.error || fallbackMessage);
      return payload;
    }

    function submissionNode(app) {
      const wrapper = document.createElement("article");
      wrapper.className = "submission";

      const title = document.createElement("strong");
      title.textContent = text(app.appName) || "Untitled submission";
      wrapper.append(title);

      const meta = document.createElement("div");
      meta.className = "meta";
      meta.textContent = text(app.teamMembers) || "Team pending";
      wrapper.append(meta);

      const links = document.createElement("div");
      links.className = "links";
      const appLink = document.createElement("a");
      appLink.href = app.liveUrl;
      appLink.target = "_blank";
      appLink.rel = "noopener";
      appLink.textContent = "Open app";
      links.append(appLink);
      const videoLink = document.createElement("a");
      videoLink.href = app.videoUrl;
      videoLink.target = "_blank";
      videoLink.rel = "noopener";
      videoLink.textContent = "Pitch video";
      links.append(videoLink);
      wrapper.append(links);

      const submitted = document.createElement("div");
      submitted.textContent = `Submitted: ${text(app.submittedAt)}`;
      wrapper.append(submitted);

      const deleteButton = document.createElement("button");
      deleteButton.className = "danger";
      deleteButton.type = "button";
      deleteButton.textContent = "Delete submission";
      deleteButton.addEventListener("click", async () => {
        const pin = pinInputValue();
        if (!pin) {
          setStatus("Enter the 4-digit admin PIN first.", true);
          return;
        }
        if (!confirm(`Delete "${title.textContent}" from the repository?`)) return;
        deleteButton.disabled = true;
        try {
          const response = await fetch(`/submissions/${encodeURIComponent(app.id)}`, {
            method: "DELETE",
            headers: { Authorization: `Bearer ${pin}` }
          });
          await jsonOrError(response, "Delete failed.");
          wrapper.remove();
          setStatus("Submission deleted.");
        } catch (error) {
          deleteButton.disabled = false;
          setStatus(error.message || "Delete failed.", true);
        }
      });
      wrapper.append(deleteButton);

      return wrapper;
    }

    function renderVoteResults(payload) {
      const totals = Array.isArray(payload.totals) ? payload.totals : [];
      const votes = Array.isArray(payload.votes) ? payload.votes : [];
      voteResultsBox.textContent = "";

      if (!totals.length && !votes.length) {
        voteResultsBox.textContent = "No votes yet.";
        return;
      }

      const totalsSection = document.createElement("section");
      const totalsTitle = document.createElement("strong");
      totalsTitle.textContent = "Totals";
      totalsSection.append(totalsTitle);

      const table = document.createElement("table");
      const thead = document.createElement("thead");
      thead.innerHTML = "<tr><th>Submission</th><th>Votes</th></tr>";
      table.append(thead);

      const tbody = document.createElement("tbody");
      totals.forEach((total) => {
        const row = document.createElement("tr");
        const appCell = document.createElement("td");
        appCell.textContent = text(total.appName) || "Deleted submission";
        const countCell = document.createElement("td");
        countCell.textContent = String(total.count || 0);
        row.append(appCell, countCell);
        tbody.append(row);
      });
      table.append(tbody);
      totalsSection.append(table);
      voteResultsBox.append(totalsSection);

      const ballotsSection = document.createElement("section");
      const ballotsTitle = document.createElement("strong");
      ballotsTitle.textContent = "Ballots";
      ballotsSection.append(ballotsTitle);

      const ballotList = document.createElement("div");
      ballotList.className = "ballots";
      votes.forEach((vote) => {
        const ballot = document.createElement("article");
        ballot.className = "ballot";
        const uid = document.createElement("strong");
        uid.textContent = `UID ${text(vote.studentUid)}`;
        ballot.append(uid);

        const choices = document.createElement("ol");
        (Array.isArray(vote.choices) ? vote.choices : []).forEach((choice) => {
          const item = document.createElement("li");
          item.textContent = text(choice.appName) || "Deleted submission";
          choices.append(item);
        });
        ballot.append(choices);
        ballotList.append(ballot);
      });
      ballotsSection.append(ballotList);
      voteResultsBox.append(ballotsSection);
    }

    async function loadAdminData() {
      const pin = pinInputValue();
      if (!pin) {
        setStatus("Enter the 4-digit admin PIN first.", true);
        return;
      }

      setStatus("Loading...");
      submissionsBox.textContent = "";
      voteResultsBox.textContent = "";
      try {
        const [submissionPayload, votePayload] = await Promise.all([
          fetch("/submissions").then((response) => jsonOrError(response, "Could not load submissions.")),
          fetch("/votes", { headers: { Authorization: `Bearer ${pin}` } })
            .then((response) => jsonOrError(response, "Could not load voting results."))
        ]);
        const submissions = Array.isArray(submissionPayload.submissions) ? submissionPayload.submissions : [];
        submissions.forEach((app) => submissionsBox.append(submissionNode(app)));
        renderVoteResults(votePayload);
        setStatus(`${submissions.length} submissions and ${Array.isArray(votePayload.votes) ? votePayload.votes.length : 0} ballots loaded.`);
      } catch (error) {
        setStatus(error.message || "Could not load admin data.", true);
      }
    }

    function pinInputValue() {
      return pinInput.value.trim();
    }

    loadButton.addEventListener("click", loadAdminData);
  </script>
</body>
</html>"""
    return page.encode("utf-8")


class AppGalleryHandler(BaseHTTPRequestHandler):
    server_version = "SSAIAppGallery/1.0"

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", os.environ.get("ALLOWED_ORIGIN", "*"))
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, X-Admin-Token")
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
            self.write_json(
                HTTPStatus.OK,
                {"ok": True, "adminConfigured": bool(configured_admin_pin()), "database": str(DB_PATH)},
            )
            return

        if parsed.path == "/submissions":
            self.write_json(HTTPStatus.OK, {"submissions": list_submissions()})
            return

        if parsed.path == "/votes":
            if not configured_admin_pin():
                self.write_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "Admin voting results are not configured."})
                return

            if not admin_pin_is_valid(self.headers):
                self.write_json(HTTPStatus.UNAUTHORIZED, {"error": "Invalid admin PIN."})
                return

            try:
                self.write_json(HTTPStatus.OK, vote_results())
            except sqlite3.Error:
                self.write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Vote repository is unavailable."})
            return

        if parsed.path == "/":
            body = render_repository_page(list_submissions())
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/admin":
            body = render_admin_page()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.write_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path not in {"/submissions", "/votes"}:
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
            if parsed.path == "/submissions":
                submission = save_submission(validate_submission(payload))
                self.write_json(HTTPStatus.CREATED, {"submission": submission})
            else:
                vote = save_vote(validate_vote(payload))
                self.write_json(HTTPStatus.CREATED, {"vote": vote})
        except json.JSONDecodeError:
            self.write_json(HTTPStatus.BAD_REQUEST, {"error": "Request body must be valid JSON."})
            return
        except ValueError as error:
            self.write_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        except sqlite3.Error:
            repository_name = "Vote" if parsed.path == "/votes" else "Submission"
            self.write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{repository_name} repository is unavailable."})
            return

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        match = re.fullmatch(r"/submissions/([^/]+)", parsed.path)
        if not match:
            self.write_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
            return

        if not configured_admin_pin():
            self.write_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "Admin deletion is not configured."})
            return

        if not admin_pin_is_valid(self.headers):
            self.write_json(HTTPStatus.UNAUTHORIZED, {"error": "Invalid admin PIN."})
            return

        submission_id = match.group(1)
        if not ID_PATTERN.fullmatch(submission_id):
            self.write_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid submission id."})
            return

        try:
            deleted = delete_submission(submission_id)
        except sqlite3.Error:
            self.write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Submission repository is unavailable."})
            return

        if not deleted:
            self.write_json(HTTPStatus.NOT_FOUND, {"error": "Submission not found."})
            return

        self.write_json(HTTPStatus.OK, {"deleted": True, "id": submission_id})


def main() -> None:
    ensure_database()
    port = int(os.environ.get("PORT", "8787"))
    server = ThreadingHTTPServer(("0.0.0.0", port), AppGalleryHandler)
    print(f"SSAI app gallery repository listening on port {port}")
    print(f"Database: {DB_PATH}")
    server.serve_forever()


if __name__ == "__main__":
    main()
