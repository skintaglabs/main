import { useEffect, useState } from 'react'

interface RiskDisplayProps {
  score: number
  tier: 'low' | 'moderate' | 'high'
}

const tierColors = {
  low: 'var(--color-green)',
  moderate: 'var(--color-amber)',
  high: 'var(--color-red)'
}

export function RiskDisplay({ score, tier }: RiskDisplayProps) {
  const [animatedScore, setAnimatedScore] = useState(0)
  const percentage = Math.round(score * 100)
  const color = tierColors[tier]

  useEffect(() => {
    const timeout = setTimeout(() => {
      setAnimatedScore(percentage)
    }, 100)
    return () => clearTimeout(timeout)
  }, [percentage])

  return (
    <div>
      <h3 className="text-[17px] font-semibold mb-4">Risk Assessment</h3>
      <div className="flex items-end gap-2 mb-4">
        <div
          className="text-[64px] leading-none font-semibold"
          style={{ fontFamily: "'Instrument Serif', serif", color }}
        >
          {animatedScore}%
        </div>
        <div className="text-[17px] text-[var(--color-text-muted)] pb-2">
          melanoma risk
        </div>
      </div>
      <div className="h-2 bg-[var(--color-surface-alt)] rounded-full overflow-hidden">
        <div
          className="h-full transition-all duration-[1000ms]"
          style={{
            width: `${animatedScore}%`,
            backgroundColor: color,
            transitionTimingFunction: 'var(--ease-spring)'
          }}
        />
      </div>
    </div>
  )
}
