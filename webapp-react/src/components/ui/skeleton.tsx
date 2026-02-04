import { type HTMLAttributes } from 'react'
import { cn } from '@/lib/utils'

function Skeleton({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn('animate-shimmer rounded-[var(--radius)]', className)}
      {...props}
    />
  )
}

export { Skeleton }
