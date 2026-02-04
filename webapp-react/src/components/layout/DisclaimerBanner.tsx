import { AlertTriangle } from 'lucide-react'

export function DisclaimerBanner() {
  return (
    <div className="bg-[var(--color-red-bg)] border border-[var(--color-red)] rounded-[var(--radius-lg)] p-4 flex gap-3">
      <AlertTriangle className="w-5 h-5 text-[var(--color-red)] flex-shrink-0 mt-0.5" />
      <div className="text-[15px] leading-relaxed">
        <strong className="font-semibold text-[var(--color-red)]">Medical Disclaimer:</strong>{' '}
        <span className="text-[var(--color-text-secondary)]">
          This tool is for educational purposes only and should not replace professional medical advice.
          Always consult a healthcare provider for skin concerns.
        </span>
      </div>
    </div>
  )
}
