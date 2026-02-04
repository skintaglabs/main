import { ArrowLeft } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Results } from '@/components/results/Results'
import { HistoryList } from './HistoryList'
import type { HistoryEntry } from '@/hooks/useAnalysisHistory'
import { useState } from 'react'

export function HistoryView() {
  const [selectedEntry, setSelectedEntry] = useState<HistoryEntry | null>(null)

  if (selectedEntry) {
    return (
      <div className="space-y-4">
        <div className="flex items-center gap-3">
          <Button
            onClick={() => setSelectedEntry(null)}
            variant="ghost"
            size="icon"
            className="rounded-full"
          >
            <ArrowLeft className="w-5 h-5" />
          </Button>
          <div>
            <h2 className="text-[20px] font-semibold">
              {selectedEntry.fileName}
            </h2>
            <p className="text-[13px] text-[var(--color-text-muted)]">
              {new Date(selectedEntry.timestamp).toLocaleString()}
            </p>
          </div>
        </div>

        <div className="rounded-[var(--radius-lg)] overflow-hidden border bg-[var(--color-surface)]">
          <img
            src={selectedEntry.imageUrl}
            alt="Analysis"
            className="w-full aspect-[4/3] object-contain bg-[var(--color-surface-alt)]"
          />
        </div>

        <Results results={selectedEntry.results} />
      </div>
    )
  }

  return (
    <HistoryList onViewEntry={setSelectedEntry} />
  )
}
