import { Button } from '@/components/ui/button'
import { MapPin, Download, Share2 } from 'lucide-react'
import { toast } from 'sonner'

interface CTAActionsProps {
  tier: 'low' | 'moderate' | 'high'
}

const tierActions = {
  low: {
    primary: 'Monitor and Track',
    secondary: 'Learn More About Prevention'
  },
  moderate: {
    primary: 'Find Dermatologists',
    secondary: 'Schedule Consultation'
  },
  high: {
    primary: 'Find Dermatologists Urgently',
    secondary: 'Save and Share Results'
  }
}

export function CTAActions({ tier }: CTAActionsProps) {
  const actions = tierActions[tier]

  const handleClick = (action: string) => {
    toast.info('Feature coming soon!', {
      description: `${action} will be available in a future update.`
    })
  }

  return (
    <div className="space-y-3">
      <Button
        onClick={() => handleClick(actions.primary)}
        className="w-full"
        size="lg"
      >
        <MapPin className="w-5 h-5" />
        {actions.primary}
      </Button>

      <div className="grid grid-cols-2 gap-3">
        <Button
          onClick={() => handleClick('Save Results')}
          variant="outline"
          size="default"
        >
          <Download className="w-4 h-4" />
          Save
          <span className="ml-auto text-[11px] px-1.5 py-0.5 rounded-full bg-[var(--color-surface-alt)] border">
            Soon
          </span>
        </Button>
        <Button
          onClick={() => handleClick('Share Results')}
          variant="outline"
          size="default"
        >
          <Share2 className="w-4 h-4" />
          Share
          <span className="ml-auto text-[11px] px-1.5 py-0.5 rounded-full bg-[var(--color-surface-alt)] border">
            Soon
          </span>
        </Button>
      </div>
    </div>
  )
}
