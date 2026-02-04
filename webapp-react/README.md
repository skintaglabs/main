# SkinTag React Application

Modern React implementation of the SkinTag web interface, built with Vite and TypeScript.

## Tech Stack

- React 19
- TypeScript 5.9
- Vite 7
- Tailwind CSS 4
- Radix UI
- Sonner (Toast notifications)

## Development

```bash
npm install
npm run dev
```

## Build

```bash
npm run build
npm run preview
```

## Configuration

### API Server URL

The frontend connects to the inference server. Configure via (in priority order):

1. **URL parameter**: `?api=https://your-server.com`
2. **Environment variable**: `VITE_API_URL`
3. **Default**: `http://localhost:8000`

**Local development with custom API:**

```bash
# Create .env file
cp .env.example .env

# Edit .env
VITE_API_URL=https://your-tunnel-url.trycloudflare.com

# Run dev server
npm run dev
```

**One-time override:**

```bash
VITE_API_URL=https://custom-url.com npm run dev
```

## Architecture

- `src/components/ui/` - Radix UI wrappers (Button, Card, Sheet, Skeleton)
- `src/components/upload/` - Upload zone, camera, preview
- `src/components/results/` - Results display components
- `src/components/layout/` - Header, footer, disclaimer
- `src/hooks/` - Custom hooks (validation, network, analysis)
- `src/contexts/` - App state management
- `src/lib/` - Utilities and API client
- `src/types/` - TypeScript interfaces

## Design System

Preserves SkinTag's warm beige aesthetic with iOS design tokens:
- 8pt grid spacing
- 16px/24px border radius
- 3-tier shadow system
- Instrument Serif for headings
- DM Sans for body text
