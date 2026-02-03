# Serverless Inference

Run inference in GitHub Actions via Cloudflare Tunnel - no server needed.

## Setup

```bash
# Add to .env
echo "CLOUDFLARE_API_TOKEN=your_token" > .env
echo "CLOUDFLARE_ACCOUNT_ID=your_account_id" >> .env

# Run setup
./scripts/setup-tunnel.sh
```

Get from:
- API token: https://dash.cloudflare.com/profile/api-tokens
- Account ID: https://dash.cloudflare.com/ (in sidebar)

**Note:** HF_TOKEN optional (public models work without it)

Then: Actions → Deploy Inference Server → Run workflow

## URL

Tunnel URL: `https://<tunnel-id>.cfargotunnel.com`

The setup script will output the full URL. Configure your frontend to use this as the API endpoint.

## Features

- Free (2000 min/month)
- Auto-restart every 5.5h
- ~1-2 min cold start (optimized with uv)

## Limits

- 6-hour max runtime
- 30-60s downtime on restart
- Single concurrent user
