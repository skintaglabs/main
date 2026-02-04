import { useAppContext } from '@/contexts/AppContext'
import { useNetworkStatus } from './useNetworkStatus'
import { analyzeImage } from '@/lib/api'
import { API_URL } from '@/config/api'
import { toast } from 'sonner'

export function useAnalysis() {
  const { setIsAnalyzing, setResults } = useAppContext()
  const { isOnline } = useNetworkStatus()

  const analyze = async (file: File) => {
    if (!isOnline) {
      toast.error('No internet connection')
      return
    }

    setIsAnalyzing(true)

    const slowTimeout = setTimeout(() => {
      toast.info('Still analyzing...', { duration: Infinity })
    }, 10000)

    try {
      const result = await analyzeImage(file, API_URL)
      clearTimeout(slowTimeout)
      toast.dismiss()
      setResults(result)
      toast.success('Analysis complete', { duration: 2000 })
    } catch (error) {
      clearTimeout(slowTimeout)
      toast.dismiss()

      if (error instanceof Error && error.message === 'REQUEST_TIMEOUT') {
        toast.error('Request timed out. Try a smaller image.', {
          action: {
            label: 'Retry',
            onClick: () => analyze(file)
          }
        })
      } else {
        toast.error('Unable to analyze image')
      }
    } finally {
      setIsAnalyzing(false)
    }
  }

  return { analyze }
}
