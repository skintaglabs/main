import { toast } from 'sonner'

export function useImageValidation() {
  const validateImage = async (file: File): Promise<boolean> => {
    if (!file.type.startsWith('image/')) {
      toast.error('Please select an image file')
      return false
    }

    if (file.size > 10 * 1024 * 1024) {
      toast.error('Image must be smaller than 10MB')
      return false
    }

    if (file.size > 5 * 1024 * 1024) {
      toast.warning('Large image detected. Analysis may take longer.')
    }

    try {
      const image = new Image()
      const url = URL.createObjectURL(file)

      await new Promise<void>((resolve, reject) => {
        image.onload = () => {
          URL.revokeObjectURL(url)

          if (image.width > 4000 || image.height > 4000) {
            toast.warning('Very large image. Consider resizing for faster analysis.')
          }

          resolve()
        }
        image.onerror = () => {
          URL.revokeObjectURL(url)
          reject(new Error('Corrupted image'))
        }
        image.src = url
      })

      return true
    } catch {
      toast.error('Unable to load image. File may be corrupted.')
      return false
    }
  }

  return { validateImage }
}
