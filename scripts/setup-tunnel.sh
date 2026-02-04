#!/bin/bash
set -e

# Automated Cloudflare Tunnel setup for SkinTag
# Uses Cloudflare API (no browser login required)

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
    echo "Add to .env file:"
    echo "  echo 'CLOUDFLARE_API_TOKEN=your_token' > .env"
    echo ""
    echo "Get token from: https://dash.cloudflare.com/profile/api-tokens"
    echo "Required permission: Account: Cloudflare Tunnel (Edit)"
    exit 1
fi

# Check for Cloudflare Account ID
if [ -z "$CLOUDFLARE_ACCOUNT_ID" ]; then
    echo "Error: CLOUDFLARE_ACCOUNT_ID not found"
    echo ""
    echo "Add to .env file:"
    echo "  echo 'CLOUDFLARE_ACCOUNT_ID=your_account_id' >> .env"
    echo ""
    echo "Find your account ID at: https://dash.cloudflare.com/"
    exit 1
fi

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "Installing jq..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install jq
    else
        sudo apt-get update && sudo apt-get install -y jq
    fi
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

# Check if tunnel exists via API
echo "Checking for existing tunnel..."
EXISTING_TUNNEL=$(curl -s -X GET \
    "https://api.cloudflare.com/client/v4/accounts/$CLOUDFLARE_ACCOUNT_ID/cfd_tunnel" \
    -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
    -H "Content-Type: application/json" | jq -r ".result[] | select(.name == \"$TUNNEL_NAME\") | .id")

if [ -n "$EXISTING_TUNNEL" ]; then
    echo "Tunnel '$TUNNEL_NAME' already exists (ID: $EXISTING_TUNNEL)"
    read -p "Delete and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Deleting existing tunnel..."
        curl -s -X DELETE \
            "https://api.cloudflare.com/client/v4/accounts/$CLOUDFLARE_ACCOUNT_ID/cfd_tunnel/$EXISTING_TUNNEL" \
            -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" > /dev/null
        EXISTING_TUNNEL=""
    else
        TUNNEL_ID="$EXISTING_TUNNEL"
    fi
fi

# Create tunnel if it doesn't exist
if [ -z "$TUNNEL_ID" ]; then
    echo "Creating tunnel '$TUNNEL_NAME' via API..."

    # Generate a random secret for the tunnel
    TUNNEL_SECRET=$(openssl rand -base64 32)

    CREATE_RESPONSE=$(curl -s -X POST \
        "https://api.cloudflare.com/client/v4/accounts/$CLOUDFLARE_ACCOUNT_ID/cfd_tunnel" \
        -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
        -H "Content-Type: application/json" \
        --data "{\"name\":\"$TUNNEL_NAME\",\"tunnel_secret\":\"$TUNNEL_SECRET\"}")

    TUNNEL_ID=$(echo "$CREATE_RESPONSE" | jq -r '.result.id')

    if [ -z "$TUNNEL_ID" ] || [ "$TUNNEL_ID" = "null" ]; then
        echo "Error: Failed to create tunnel"
        echo "$CREATE_RESPONSE" | jq
        exit 1
    fi

    echo "✓ Tunnel created (ID: $TUNNEL_ID)"
fi

# Get tunnel token
echo ""
echo "Getting tunnel token..."
TOKEN_RESPONSE=$(curl -s -X GET \
    "https://api.cloudflare.com/client/v4/accounts/$CLOUDFLARE_ACCOUNT_ID/cfd_tunnel/$TUNNEL_ID/token" \
    -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN")

TUNNEL_TOKEN=$(echo "$TOKEN_RESPONSE" | jq -r '.result')

if [ -z "$TUNNEL_TOKEN" ] || [ "$TUNNEL_TOKEN" = "null" ]; then
    echo "Error: Failed to get tunnel token"
    echo "$TOKEN_RESPONSE" | jq
    exit 1
fi

# Set GitHub secrets
echo ""
echo "Setting GitHub secrets..."

if ! command -v gh &> /dev/null; then
    echo "GitHub CLI not found. Install with: brew install gh"
    echo ""
    echo "Add this secret manually:"
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

# Get tunnel info to show URL
echo ""
echo "Getting tunnel URL..."
TUNNEL_INFO=$(curl -s -X GET \
    "https://api.cloudflare.com/client/v4/accounts/$CLOUDFLARE_ACCOUNT_ID/cfd_tunnel/$TUNNEL_ID" \
    -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN")

TUNNEL_DOMAIN=$(echo "$TUNNEL_INFO" | jq -r '.result.id + ".cfargotunnel.com"')

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Tunnel: $TUNNEL_NAME (ID: $TUNNEL_ID)"
echo "URL: https://$TUNNEL_DOMAIN"
echo "GitHub secrets configured"
echo ""
echo "Configure your frontend to use this API URL:"
echo "  https://$TUNNEL_DOMAIN"
echo ""
echo "Next steps:"
echo "  1. Go to Actions → Deploy Inference Server"
echo "  2. Run workflow"
echo "  3. Frontend will be available at the URL above"
echo ""
