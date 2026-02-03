# Serverless Inference

Run inference in GitHub Actions via Cloudflare Tunnel - no server needed.

## Setup

1. **Create Cloudflare Tunnel:**
   ```bash
   brew install cloudflare/cloudflare/cloudflared
   cloudflared tunnel login
   cloudflared tunnel create skintag-inference
   cloudflared tunnel token skintag-inference  # Save this
   ```

2. **Add GitHub Secrets:**
   - `SKINTAG_TUNNEL_TOKEN` - Cloudflare token
   - `HF_TOKEN` - Hugging Face token

3. **Run Workflow:**
   Actions → Deploy Inference Server → Run workflow

4. **Configure Frontend:**
   Set `API_URL` to your tunnel subdomain

## Features

- Free (2000 min/month)
- Auto-restart every 5.5h
- ~2-3 min cold start

## Limits

- 6-hour max runtime
- 30-60s downtime on restart
- Single concurrent user
