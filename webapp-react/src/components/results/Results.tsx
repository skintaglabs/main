import type { AnalysisResult } from '@/types'
import { TierCard } from './TierCard'
import { RiskDisplay } from './RiskDisplay'
import { ABCDEGrid } from './ABCDEGrid'
import { ConditionCard } from './ConditionCard'
import { BinaryBarsCard } from './BinaryBarsCard'
import { CTAActions } from './CTAActions'
import { DisclaimerBanner } from '@/components/layout/DisclaimerBanner'
import { useAppContext } from '@/contexts/AppContext'

interface ResultsProps {
  results: AnalysisResult
  onAnalyzeAnother?: () => void
}

export function Results({ results, onAnalyzeAnother }: ResultsProps) {
  const { state } = useAppContext()

  return (
    <div className="space-y-6">
      <div id="results-capture" className="space-y-6">
        {state.previewUrl && (
          <div className="flex justify-center">
            <img
              src={state.previewUrl}
              alt="Analyzed image"
              className="w-48 h-48 object-cover rounded-lg border-2 border-[var(--color-border)]"
            />
          </div>
        )}

        <TierCard
          tier={results.urgency_tier}
          confidence={results.confidence}
          recommendation={results.recommendation}
        />

        <RiskDisplay
          score={results.risk_score}
          tier={results.urgency_tier}
        />

        <ABCDEGrid />

        {results.condition_estimate && results.condition_probabilities && (
          <ConditionCard
            topCondition={results.condition_estimate}
            conditions={results.condition_probabilities}
          />
        )}

        <BinaryBarsCard
          benign={results.probabilities.benign}
          malignant={results.probabilities.malignant}
        />

        <DisclaimerBanner />
      </div>

      <CTAActions tier={results.urgency_tier} results={results} onAnalyzeAnother={onAnalyzeAnother} />
    </div>
  )
}
