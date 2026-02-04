import { Camera } from 'lucide-react'
import { type ChangeEvent } from 'react'

interface CameraButtonProps {
  onCapture: (file: File) => void
}

export function CameraButton({ onCapture }: CameraButtonProps) {
  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      onCapture(file)
    }
  }

  return (
    <div className="block sm:hidden">
      <input
        type="file"
        accept="image/*"
        capture="environment"
        onChange={handleChange}
        className="hidden"
        id="camera-input"
      />
      <label htmlFor="camera-input" className="block">
        <div className="inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-[var(--radius-full)] font-medium transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 border border-[var(--color-border)] bg-[var(--color-surface)] shadow-[var(--shadow-sm)] hover:shadow-[var(--shadow)] active:scale-[0.98] h-[56px] px-8 text-[19px] w-full cursor-pointer">
          <Camera className="w-5 h-5" />
          Take Photo
        </div>
      </label>
    </div>
  )
}
