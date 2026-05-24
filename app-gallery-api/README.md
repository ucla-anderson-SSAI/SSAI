# SSAI App Gallery Repository

This is a tiny Railway-hosted submission repository for `app-library.html`.
It uses only Python standard library code and a SQLite database.

## Railway Setup

From this folder:

```bash
railway init --name ssai-app-gallery-api
railway up --detach
railway volume add --mount-path /data
railway domain --port 8787
```

Then put the generated Railway domain into `APP_LIBRARY_CONFIG.apiBaseUrl` in
`../app-library.html`.

Current Railway API:

```text
https://ssai-app-gallery-api-production.up.railway.app
```

The app stores data at `/data/app_submissions.sqlite3` when the Railway volume is mounted.
Without the volume, Railway's filesystem may be replaced on future deploys.

## Endpoints

- `GET /` shows a simple public HTML repository view.
- `GET /submissions` returns all submissions as JSON.
- `POST /submissions` creates a submission.
- `GET /health` checks that the service is running.

## Local Test

```bash
python server.py
```

Then open `http://localhost:8787` or call `http://localhost:8787/submissions`.
