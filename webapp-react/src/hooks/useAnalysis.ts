import { useAppContext } from '@/contexts/AppContext'
import { useNetworkStatus } from './useNetworkStatus'
import { analyzeImage } from '@/lib/api'
import { API_URL } from '@/config/api'
import { toast } from 'sonner'
import type { AnalysisResult } from '@/types'

export function useAnalysis() {
  const { setIsAnalyzing, setResults } = useAppContext()
  const { isOnline } = useNetworkStatus()

  const analyze = async (file: File): Promise<AnalysisResult | null> => {
    if (!isOnline) {
      toast.error('No internet connection')
      return null
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
      return result
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
        toast.error('Unable to connect to API server', {
          description: 'The server may have restarted. Try refreshing the page.',
          duration: 5000,
          action: {
            label: 'Refresh',
            onClick: () => window.location.reload()
          }
        })
      }
      return null
    } finally {
      setIsAnalyzing(false)
    }
  }

  return { analyze }
}
