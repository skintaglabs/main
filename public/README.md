# Frontend Versions

Two frontend versions deployed to GitHub Pages.

## Structure

```
/                # Landing page (version selector)
/classic/        # Simple HTML (no build, fast)
/app/            # Next.js (animations, camera)
```

## Local Dev

```bash
# Classic
python -m http.server 8080 --directory public

# Modern
cd app/skintag-webapp && npm run dev
```

## API Config

**Classic**: URL param `?api=https://...` or `config.js`
**Modern**: `NEXT_PUBLIC_API_URL` env variable

Deployment auto-configured via workflow.
