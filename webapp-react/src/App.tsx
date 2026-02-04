import { Toaster } from 'sonner'
import { AppProvider, useAppContext } from '@/contexts/AppContext'
import { useImageValidation } from '@/hooks/useImageValidation'
import { useAnalysis } from '@/hooks/useAnalysis'
import { Header } from '@/components/layout/Header'
import { Footer } from '@/components/layout/Footer'
import { DisclaimerBanner } from '@/components/layout/DisclaimerBanner'
import { OnboardingModal } from '@/components/layout/OnboardingModal'
import { UploadZone } from '@/components/upload/UploadZone'
import { CameraButton } from '@/components/upload/CameraButton'
import { PreviewCard } from '@/components/upload/PreviewCard'
import { ResultsContainer } from '@/components/results/ResultsContainer'
import { SkeletonResults } from '@/components/results/SkeletonResults'
import { Results } from '@/components/results/Results'

function AppContent() {
  const { state, setSelectedFile, clearImage, setShowResults } = useAppContext()
  const { validateImage } = useImageValidation()
  const { analyze } = useAnalysis()

  const handleFileSelect = (file: File, previewUrl: string) => {
    setSelectedFile(file, previewUrl)
  }

  const handleCameraCapture = async (file: File) => {
    const isValid = await validateImage(file)
    if (isValid) {
      const previewUrl = URL.createObjectURL(file)
      setSelectedFile(file, previewUrl)
    }
  }

  const handleAnalyze = () => {
    if (state.selectedFile) {
      analyze(state.selectedFile)
    }
  }

  const handleCloseResults = () => {
    setShowResults(false)
  }

  return (
    <div className="min-h-screen flex flex-col">
      <OnboardingModal />

      <div className="flex-1 w-full max-w-3xl mx-auto px-4">
        <Header />

        <main className="pb-8">
          <div className="mb-6">
            <DisclaimerBanner />
          </div>

          {!state.selectedFile && (
            <div className="space-y-4">
              <UploadZone onFileSelect={handleFileSelect} />
              <CameraButton onCapture={handleCameraCapture} />
            </div>
          )}

          {state.selectedFile && state.previewUrl && !state.isAnalyzing && !state.showResults && (
            <PreviewCard
              file={state.selectedFile}
              previewUrl={state.previewUrl}
              onClear={clearImage}
              onAnalyze={handleAnalyze}
            />
          )}

          {state.isAnalyzing && (
            <div className="max-w-2xl mx-auto">
              <SkeletonResults />
            </div>
          )}

          {state.results && (
            <ResultsContainer showResults={state.showResults} onClose={handleCloseResults}>
              <Results results={state.results} />
            </ResultsContainer>
          )}
        </main>
      </div>

      <Footer />

      <Toaster
        position="top-center"
        toastOptions={{
          style: {
            background: 'var(--color-surface)',
            border: '1px solid var(--color-border)',
            borderRadius: 'var(--radius)',
            boxShadow: 'var(--shadow-lg)',
            fontFamily: 'inherit'
          }
        }}
      />
    </div>
  )
}

function App() {
  return (
    <AppProvider>
      <AppContent />
    </AppProvider>
  )
}

export default App
