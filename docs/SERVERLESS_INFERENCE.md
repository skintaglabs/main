# Serverless Inference

Run inference in GitHub Actions via Cloudflare Tunnel - no server needed.

## Setup

**Automated:**
```bash
# Add to .env file
echo "CLOUDFLARE_API_TOKEN=your_token" > .env

# Run setup
./scripts/setup-tunnel.sh
```

Get token from: https://dash.cloudflare.com/profile/api-tokens

Then: Actions → Deploy Inference Server → Run workflow

## Features

- Free (2000 min/month)
- Auto-restart every 5.5h
- ~2-3 min cold start

## Limits

- 6-hour max runtime
- 30-60s downtime on restart
- Single concurrent user
