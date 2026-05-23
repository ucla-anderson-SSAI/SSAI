#!/usr/bin/env node

const repo = process.argv[2] || "ucla-anderson-SSAI/SSAI";
const format = process.argv[3] || "csv";
const marker = "<!-- app-submission-metadata-v1 -->";
const summaryFields = ["problem", "build", "endUsers", "limitations", "improvements"];

function csvCell(value) {
  const text = Array.isArray(value) ? value.join(", ") : String(value ?? "");
  return `"${text.replaceAll('"', '""')}"`;
}

function parseMetadata(body) {
  if (!body.includes(marker)) return null;
  const match = body.match(/```json\s*([\s\S]*?)\s*```/);
  if (!match) return null;

  try {
    return JSON.parse(match[1]);
  } catch (error) {
    return null;
  }
}

async function fetchIssues(page = 1) {
  const params = new URLSearchParams({
    state: "all",
    per_page: "100",
    page: String(page)
  });
  const response = await fetch(`https://api.github.com/repos/${repo}/issues?${params}`, {
    headers: {
      Accept: "application/vnd.github+json",
      "User-Agent": "ssai-app-submission-export"
    }
  });

  if (!response.ok) {
    throw new Error(`GitHub API request failed (${response.status}): ${await response.text()}`);
  }

  return response.json();
}

async function main() {
  const submissions = [];

  for (let page = 1; ; page += 1) {
    const issues = await fetchIssues(page);
    if (!issues.length) break;

    for (const issue of issues) {
      if (issue.pull_request) continue;
      const metadata = parseMetadata(issue.body || "");
      if (!metadata) continue;
      submissions.push({
        issueNumber: issue.number,
        issueUrl: issue.html_url,
        issueState: issue.state,
        appName: metadata.appName,
        teamMembers: metadata.teamMembers,
        liveUrl: metadata.liveUrl,
        videoUrl: metadata.videoUrl,
        keywords: metadata.keywords,
        problem: metadata.summary?.problem || metadata.problem || "",
        build: metadata.summary?.build || metadata.build || "",
        endUsers: metadata.summary?.endUsers || metadata.endUsers || "",
        limitations: metadata.summary?.limitations || metadata.limitations || "",
        improvements: metadata.summary?.improvements || metadata.improvements || "",
        writeup: metadata.writeup,
        submittedAt: metadata.submittedAt,
        createdAt: issue.created_at
      });
    }
  }

  if (format === "json") {
    console.log(JSON.stringify(submissions, null, 2));
    return;
  }

  const headers = [
    "issueNumber",
    "issueUrl",
    "issueState",
    "appName",
    "teamMembers",
    "liveUrl",
    "videoUrl",
    "keywords",
    ...summaryFields,
    "writeup",
    "submittedAt",
    "createdAt"
  ];
  console.log(headers.join(","));
  for (const submission of submissions) {
    console.log(headers.map((header) => csvCell(submission[header])).join(","));
  }
}

main().catch((error) => {
  console.error(error.message);
  process.exit(1);
});
