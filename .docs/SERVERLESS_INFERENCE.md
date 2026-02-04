# Serverless Inference

Run inference in GitHub Actions via Cloudflare Tunnel - no setup needed.

## Usage

Just run the workflow: Actions → Deploy Inference Server → Run workflow

The tunnel URL is automatically published to the frontend.

**Note:** HF_TOKEN optional (public models work without it)

## Features

- Free (2000 min/month)
- Auto-restart every 5.5h
- ~1-2 min cold start (optimized with uv)

## Limits

- 6-hour max runtime
- 30-60s downtime on restart
- Single concurrent user
