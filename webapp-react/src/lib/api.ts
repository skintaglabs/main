import type { AnalysisResult } from '@/types'

export async function analyzeImage(file: File, apiUrl: string): Promise<AnalysisResult> {
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), 60000)

  const formData = new FormData()
  formData.append('file', file)

  try {
    const response = await fetch(`${apiUrl}/api/analyze`, {
      method: 'POST',
      body: formData,
      signal: controller.signal
    })

    clearTimeout(timeoutId)

    if (!response.ok) {
      const error = await response.json().catch(() => ({}))
      throw new Error(error.detail || 'Analysis failed')
    }

    return await response.json()
  } catch (error) {
    clearTimeout(timeoutId)
    if (error instanceof Error && error.name === 'AbortError') {
      throw new Error('REQUEST_TIMEOUT')
    }
    throw error
  }
}
