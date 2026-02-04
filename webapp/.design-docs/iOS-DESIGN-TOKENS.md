# iOS Design Tokens Implementation

## Overview
iOS design tokens have been implemented in `webapp/index.html` to ensure the interface feels native to iOS while preserving the warm, classic aesthetic.

## Design Tokens Added

### 8pt Grid System
- `--space-1` through `--space-8` (8px, 16px, 24px, 32px, 40px, 48px, 56px, 64px)
- Applied to padding, margins, gaps throughout the interface

### iOS Typography Scale
- `--text-xs: 11px` - Labels, badges, fine print
- `--text-sm: 13px` - Secondary text, descriptions
- `--text-base: 15px` - Body text (iOS default)
- `--text-lg: 17px` - Emphasized body text
- `--text-xl: 20px` - Subheadings
- `--text-2xl: 28px` - Section titles
- `--text-3xl: 34px` - Page titles (mobile)
- `--text-4xl: 40px` - Page titles (desktop)

### 8pt-aligned Line Heights
- `--leading-tight: 24px` - Compact text
- `--leading-normal: 32px` - Body text
- `--leading-relaxed: 40px` - Comfortable reading
- `--leading-loose: 48px` - Headers

### Touch Targets
- `--touch-target: 44px` - iOS standard minimum for interactive elements
- Applied to all buttons, badges, and interactive controls

### iOS Animation Timing
- `--ease-in-out: cubic-bezier(0.42, 0, 0.58, 1)` - Standard easing
- `--ease-out: cubic-bezier(0.25, 0.1, 0.25, 1)` - Smooth deceleration
- `--ease-spring: cubic-bezier(0.22, 1, 0.36, 1)` - Springy, natural motion
- `--duration-fast: 0.2s` - Quick interactions
- `--duration-normal: 0.3s` - Standard transitions
- `--duration-slow: 0.5s` - Elaborate animations

## Key Changes

### Spacing & Layout
- Page padding: `var(--space-4) var(--space-3) var(--space-8)`
- Masthead margin: `var(--space-5)`
- Card padding: `var(--space-3)`
- Consistent gap sizing using 8pt grid

### Typography
- Body font size: `var(--text-base)` (15px)
- Body line height: `var(--leading-normal)` (32px)
- Logo: `var(--text-4xl)` with `var(--leading-loose)`
- All text elements use iOS typography scale

### Interactive Elements
- All buttons have `min-height: var(--touch-target)` (44px)
- Upload zone has adequate sizing for touch interaction
- Buttons use `flex` alignment for consistent touch targets

### Animations
- Upload zone: `var(--duration-normal) var(--ease-out)`
- Button hover/active: `var(--duration-fast) var(--ease-out)`
- Results animation: `var(--duration-slow) var(--ease-spring)`
- Progress bars: `var(--ease-spring)` for natural feel

## Preserved Features
- Warm color palette (#f6f4f0 background, earthy tones)
- Instrument Serif + DM Sans fonts
- Noise texture overlay
- Dark mode support
- High contrast mode
- Accessibility features
- Haptic feedback
- All visual character intact

## Usage Count
44 instances of iOS design tokens throughout the stylesheet

## Benefits
1. **Consistency**: All spacing follows 8pt grid
2. **Touch-Friendly**: 44px minimum touch targets
3. **Readable**: iOS-standard typography scale
4. **Natural Motion**: Spring-based easing curves
5. **Maintainable**: Centralized token system
6. **Accessible**: Preserved all accessibility features
