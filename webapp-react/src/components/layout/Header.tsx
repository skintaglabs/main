export function Header() {
  return (
    <header className="flex flex-col items-center gap-3 pt-8 pb-6">
      <h1 className="text-[40px] leading-[1.1] tracking-tight text-[var(--color-accent-warm)]" style={{ fontFamily: "'Instrument Serif', serif" }}>
        Skin<span style={{ fontStyle: 'italic' }}>Tag</span>
      </h1>
      <p className="text-[15px] text-[var(--color-text-secondary)] text-center max-w-md">
        Evaluate your skin lesion risk with advanced AI technology
      </p>
      <div className="flex items-center gap-2 mt-2 px-3 py-1.5 rounded-full bg-[var(--color-surface)] border text-[13px] text-[var(--color-text-muted)]">
        <div className="w-1.5 h-1.5 rounded-full bg-[var(--color-green)]" />
        Trained on clinical dermoscopy images
      </div>
    </header>
  )
}
