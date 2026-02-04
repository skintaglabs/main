import { type ReactNode } from 'react'
import { Sheet, SheetContent } from '@/components/ui/sheet'

interface ResultsContainerProps {
  showResults: boolean
  children: ReactNode
  onClose: () => void
}

export function ResultsContainer({ showResults, children, onClose }: ResultsContainerProps) {
  const isMobile = typeof window !== 'undefined' && window.innerWidth < 768

  if (isMobile) {
    return (
      <Sheet open={showResults} onOpenChange={onClose}>
        <SheetContent>
          <div className="overflow-y-auto -mx-6 px-6 pb-6" style={{ maxHeight: 'calc(85vh - 4rem)' }}>
            {children}
          </div>
        </SheetContent>
      </Sheet>
    )
  }

  if (!showResults) return null

  return (
    <div className="animate-fadeUp">
      {children}
    </div>
  )
}
