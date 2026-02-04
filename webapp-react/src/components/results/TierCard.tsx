interface TierCardProps {
  tier: 'low' | 'moderate' | 'high'
  confidence: string
  recommendation: string
}

const tierConfig = {
  low: {
    color: 'var(--color-green)',
    bg: 'var(--color-green-bg)',
    label: 'Low Risk'
  },
  moderate: {
    color: 'var(--color-amber)',
    bg: 'var(--color-amber-bg)',
    label: 'Moderate Risk'
  },
  high: {
    color: 'var(--color-red)',
    bg: 'var(--color-red-bg)',
    label: 'High Risk'
  }
}

export function TierCard({ tier, confidence, recommendation }: TierCardProps) {
  const config = tierConfig[tier]

  return (
    <div
      className="rounded-[var(--radius-lg)] p-6 border"
      style={{ backgroundColor: config.bg, borderColor: config.color }}
    >
      <div className="flex items-center gap-3 mb-4">
        <div
          className="w-3 h-3 rounded-full"
          style={{ backgroundColor: config.color }}
        />
        <h3 className="text-[24px] font-semibold" style={{ color: config.color }}>
          {config.label}
        </h3>
        <div className="ml-auto px-3 py-1 rounded-full bg-[var(--color-surface)] border text-[13px] font-medium">
          {confidence} confidence
        </div>
      </div>
      <p className="text-[15px] leading-relaxed text-[var(--color-text-secondary)]">
        {recommendation}
      </p>
    </div>
  )
}
