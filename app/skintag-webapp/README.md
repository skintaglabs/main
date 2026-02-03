# SkinTag WebApp

Modern web interface for SkinTag skin lesion triage system.

## Features

- Camera capture and file upload
- Real-time analysis via external API
- Risk score visualization
- Condition probability estimates
- Mobile-optimized design with Framer Motion animations

## Setup

### Local Development

1. Install dependencies:
   ```bash
   npm install
   ```

2. Create `.env.local` file:
   ```bash
   cp .env.example .env.local
   ```

3. Set your API endpoint in `.env.local`:
   ```
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```

4. Run development server:
   ```bash
   npm run dev
   ```

   Open [http://localhost:3000](http://localhost:3000)

### Build for Production

```bash
npm run build
```

Static files will be output to the `out/` directory.

## GitHub Pages Deployment

The app is configured to deploy automatically to GitHub Pages when changes are pushed to `main`.

### Configuration

1. **Repository Settings**: Enable GitHub Pages in repository settings
   - Source: GitHub Actions

2. **API URL Secret**: Add your inference API URL
   - Go to Settings → Secrets and variables → Actions
   - Add secret: `API_URL` with your inference server URL

3. **Push to Deploy**: Workflow triggers automatically on push to `main`

### Manual Deployment

Trigger deployment manually:
- Go to Actions → Deploy Webapp to GitHub Pages → Run workflow

## Architecture

```
User Browser → GitHub Pages (Static) → External API Server → ML Model
```

- **Frontend**: Next.js static export hosted on GitHub Pages
- **Backend**: Separate inference API (FastAPI) with no authentication
- **Model**: Downloaded from GitHub Releases by backend

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `NEXT_PUBLIC_API_URL` | Inference API endpoint | Yes |

## Tech Stack

- Next.js 16
- React 19
- TypeScript
- Tailwind CSS 4
- Framer Motion
- Lucide Icons
