# ngrok Remote Access

Expose the inference server from your NVIDIA GPU machine to the internet via HTTPS tunnel.

## Setup

1. **Create ngrok account**
   - Visit https://ngrok.com/signup
   - Sign up for free account

2. **Install ngrok**
   ```bash
   brew install ngrok
   ```

3. **Authenticate**
   - Get authtoken from https://dashboard.ngrok.com/get-started/your-authtoken
   - Run: `ngrok config add-authtoken YOUR_TOKEN`

## Usage

**Start server with public HTTPS URL:**
```bash
make app-remote
```

The terminal displays the public URL (e.g., `https://abc123.ngrok.io`)

**Stop server:**
```bash
make stop
```

## Local Access

For local-only testing:
```bash
make app
```

Server runs at http://localhost:8000
