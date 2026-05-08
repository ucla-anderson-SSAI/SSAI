# MGMT298D Course Site Upload Backend

This FastAPI app serves the root course `index.html` and accepts private PDF assignment submissions at `/submit`.

## Local Run

```bash
cd /path/to/SSAI
python -m uvicorn Apps.course_site.main:app --reload
```

Open `http://127.0.0.1:8000`.

## Production Settings

Set these environment variables in the host:

- `SUBMISSION_STORAGE_DIR`: private directory or mounted volume for submitted PDFs. Defaults to `private_submissions` in the repo.
- `SUBMISSION_ADMIN_TOKEN`: optional token for private admin listing and downloads.
- `MAX_UPLOAD_BYTES`: optional upload size limit. Defaults to 25 MB.

Admin routes are available only when `SUBMISSION_ADMIN_TOKEN` is set:

- `GET /admin/submissions?token=...`
- `GET /admin/submissions/{submission_id}/download?token=...`

Do not mount `SUBMISSION_STORAGE_DIR` as a public static directory.
