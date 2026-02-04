# Before & After Comparison: iOS Design Tokens

## Token Definitions

### Before
```css
:root {
    --radius: 12px;
    --radius-lg: 20px;
    --shadow-sm: 0 1px 2px rgba(26,23,20,0.04);
    /* No systematic spacing, typography, or timing tokens */
}
```

### After
```css
:root {
    /* iOS Design Tokens: 8pt Grid System */
    --space-1: 8px;
    --space-2: 16px;
    --space-3: 24px;
    --space-4: 32px;
    --space-5: 40px;
    --space-6: 48px;
    --space-7: 56px;
    --space-8: 64px;

    /* iOS Typography Scale */
    --text-xs: 11px;
    --text-sm: 13px;
    --text-base: 15px;
    --text-lg: 17px;
    --text-xl: 20px;
    --text-2xl: 28px;
    --text-3xl: 34px;
    --text-4xl: 40px;

    /* 8pt-aligned Line Heights */
    --leading-tight: 24px;
    --leading-normal: 32px;
    --leading-relaxed: 40px;
    --leading-loose: 48px;

    /* Touch Targets */
    --touch-target: 44px;

    /* iOS Animation Timing */
    --ease-in-out: cubic-bezier(0.42, 0, 0.58, 1);
    --ease-out: cubic-bezier(0.25, 0.1, 0.25, 1);
    --ease-spring: cubic-bezier(0.22, 1, 0.36, 1);
    --duration-fast: 0.2s;
    --duration-normal: 0.3s;
    --duration-slow: 0.5s;
}
```

## Example Updates

### Body Text
**Before:**
```css
body {
    line-height: 1.6;
    font-size: 1rem;
}
```

**After:**
```css
body {
    line-height: var(--leading-normal); /* 32px, 8pt-aligned */
    font-size: var(--text-base); /* 15px, iOS standard */
}
```

### Buttons
**Before:**
```css
.btn {
    padding: 0.55rem 1.25rem;
    transition: all 0.15s;
}
```

**After:**
```css
.btn {
    padding: 0 var(--space-3);
    min-height: var(--touch-target); /* 44px minimum */
    transition: all var(--duration-fast) var(--ease-out);
    display: inline-flex;
    align-items: center;
}
```

### Spacing
**Before:**
```css
.page {
    padding: 2rem 1.25rem 4rem; /* Arbitrary values */
}

.masthead {
    margin-bottom: 2.5rem; /* Not 8pt-aligned */
}
```

**After:**
```css
.page {
    padding: var(--space-4) var(--space-3) var(--space-8); /* 32px 24px 64px */
}

.masthead {
    margin-bottom: var(--space-5); /* 40px, 8pt-aligned */
}
```

### Animations
**Before:**
```css
.upload-zone {
    transition: all 0.25s ease; /* Generic easing */
}

.risk-meter-fill {
    transition: width 1s cubic-bezier(0.22, 1, 0.36, 1); /* Hardcoded spring */
}
```

**After:**
```css
.upload-zone {
    transition: all var(--duration-normal) var(--ease-out); /* iOS-standard */
}

.risk-meter-fill {
    transition: width 1s var(--ease-spring); /* Reusable spring */
}
```

### Typography
**Before:**
```css
.logo {
    font-size: 2.5rem; /* 40px */
    line-height: 1.1; /* Not 8pt-aligned */
}

.tagline {
    font-size: 0.875rem; /* 14px, not iOS standard */
}
```

**After:**
```css
.logo {
    font-size: var(--text-4xl); /* 40px, named scale */
    line-height: var(--leading-loose); /* 48px, 8pt-aligned */
}

.tagline {
    font-size: var(--text-sm); /* 13px, iOS standard */
    line-height: var(--leading-tight); /* 24px, 8pt-aligned */
}
```

## Impact Summary

| Aspect | Before | After |
|--------|--------|-------|
| **Spacing System** | Arbitrary rem values | 8pt grid (8px multiples) |
| **Typography** | Mixed px/rem values | iOS typography scale |
| **Line Heights** | Random ratios | 8pt-aligned (24, 32, 40, 48px) |
| **Touch Targets** | Varied, sometimes too small | Minimum 44px (iOS standard) |
| **Animations** | Hardcoded durations/easings | Named timing functions |
| **Maintainability** | Magic numbers throughout | Centralized token system |
| **iOS Feel** | Generic web | Native iOS experience |

## File Stats
- **Before:** 1135 lines
- **After:** 1228 lines (+93 lines for token definitions and updates)
- **Token Usage:** 44 instances of iOS design tokens
