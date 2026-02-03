#!/bin/bash
set -e

# Automated Cloudflare Tunnel setup for SkinTag
# Uses Cloudflare API token (no browser login required)

TUNNEL_NAME="${TUNNEL_NAME:-skintag-inference}"
REPO="${GITHUB_REPOSITORY:-MedGemma540/SkinTag}"

echo "=== SkinTag Cloudflare Tunnel Setup ==="
echo ""

# Load .env file if it exists
if [ -f .env ]; then
    echo "Loading credentials from .env..."
    export $(grep -v '^#' .env | xargs)
fi

# Check for Cloudflare API token
if [ -z "$CLOUDFLARE_API_TOKEN" ]; then
    echo "Error: CLOUDFLARE_API_TOKEN not found"
    echo ""
    echo "Option 1 - Create .env file:"
    echo "  echo 'CLOUDFLARE_API_TOKEN=your_token' > .env"
    echo ""
    echo "Option 2 - Export variable:"
    echo "  export CLOUDFLARE_API_TOKEN=your_token"
    echo ""
    echo "Get token from: https://dash.cloudflare.com/profile/api-tokens"
    echo "Required permission: Account: Cloudflare Tunnel (Edit)"
    exit 1
fi

# Check if cloudflared is installed
if ! command -v cloudflared &> /dev/null; then
    echo "Installing cloudflared..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install cloudflare/cloudflare/cloudflared
    else
        wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
        sudo mv cloudflared-linux-amd64 /usr/local/bin/cloudflared
        sudo chmod +x /usr/local/bin/cloudflared
    fi
fi

# Set API token for cloudflared
export TUNNEL_TOKEN="$CLOUDFLARE_API_TOKEN"

# Check if tunnel exists
echo "Checking for existing tunnel..."
if cloudflared tunnel list 2>/dev/null | grep -q "$TUNNEL_NAME"; then
    echo ""
    echo "Tunnel '$TUNNEL_NAME' already exists."
    read -p "Delete and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cloudflared tunnel delete "$TUNNEL_NAME"
        echo "Creating new tunnel..."
        cloudflared tunnel create "$TUNNEL_NAME"
    fi
else
    echo "Creating tunnel '$TUNNEL_NAME'..."
    cloudflared tunnel create "$TUNNEL_NAME"
fi

# Get tunnel token
echo ""
echo "Getting tunnel token..."
TUNNEL_TOKEN=$(cloudflared tunnel token "$TUNNEL_NAME")

# Set GitHub secrets
echo ""
echo "Setting GitHub secrets..."

if ! command -v gh &> /dev/null; then
    echo "GitHub CLI not found. Install with: brew install gh"
    echo ""
    echo "Add these secrets manually:"
    echo "  SKINTAG_TUNNEL_TOKEN=$TUNNEL_TOKEN"
    exit 0
fi

# Check if logged in
if ! gh auth status &> /dev/null; then
    echo "Logging in to GitHub..."
    gh auth login
fi

echo "Setting SKINTAG_TUNNEL_TOKEN..."
echo "$TUNNEL_TOKEN" | gh secret set SKINTAG_TUNNEL_TOKEN --repo="$REPO"

# Prompt for HF token
echo ""
read -sp "Enter Hugging Face token (or press Enter to skip): " HF_TOKEN
echo

if [ -n "$HF_TOKEN" ]; then
    echo "Setting HF_TOKEN..."
    echo "$HF_TOKEN" | gh secret set HF_TOKEN --repo="$REPO"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Tunnel: $TUNNEL_NAME"
echo "GitHub secrets configured"
echo ""
echo "Next steps:"
echo "  1. Go to Actions → Deploy Inference Server"
echo "  2. Run workflow"
echo "  3. Check logs for tunnel URL"
echo ""
