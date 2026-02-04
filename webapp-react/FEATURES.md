# SkinTag Feature Roadmap

Features organized from easiest to hardest implementation.

## Phase 1: Quick Wins (1-2 days each)

### 1. Download Results as Image
- Screenshot current results view
- Save as PNG with timestamp
- **Tech:** html2canvas or DOM to canvas API
- **Files:** `src/components/results/CTAActions.tsx`

### 2. Copy Results to Clipboard
- Format results as plain text
- Copy to clipboard with metadata
- **Tech:** Navigator Clipboard API
- **Files:** `src/lib/utils.ts`, `CTAActions.tsx`

### 3. Dark Mode Toggle
- Add theme context
- Extend Tailwind color variables
- Persist preference to localStorage
- **Tech:** CSS custom properties, localStorage
- **Files:** `src/contexts/ThemeContext.tsx`, `src/index.css`

### 4. Image Cropping Before Analysis
- Allow user to crop uploaded image
- Focus on specific lesion area
- **Tech:** react-easy-crop or react-image-crop
- **Files:** `src/components/upload/ImageCropper.tsx`

### 5. Tutorial/Onboarding Modal
- First-time user guide
- Explain how to take good photos
- Show example results
- **Tech:** Radix Dialog, localStorage for "seen" flag
- **Files:** `src/components/layout/OnboardingModal.tsx`

## Phase 2: Core Features (3-5 days each)

### 6. Save Analysis History (Local)
- Store past analyses in localStorage/IndexedDB
- List view of previous results
- Delete individual entries
- **Tech:** IndexedDB, date-fns for formatting
- **Files:** `src/hooks/useAnalysisHistory.ts`, `src/components/history/`

### 7. Share Results Link
- Generate shareable URL with results encoded
- Base64 encode result data in URL params
- Decode and display on load
- **Tech:** URL params, base64, compression
- **Files:** `src/lib/shareUtils.ts`, `src/hooks/useShareLink.ts`

### 8. Print-Friendly Results View
- CSS print styles
- Remove unnecessary UI elements
- Format for paper (A4/Letter)
- **Tech:** CSS @media print
- **Files:** `src/components/results/PrintView.tsx`

### 9. Export Results as PDF
- Generate PDF from results view
- Include image, risk score, recommendations
- **Tech:** jsPDF or react-pdf
- **Files:** `src/lib/pdfExport.ts`

### 10. Multi-Image Comparison
- Upload multiple images of same lesion
- Side-by-side view with results
- Track changes over time
- **Tech:** Array state management, grid layout
- **Files:** `src/components/comparison/ComparisonView.tsx`

## Phase 3: Advanced Features (1-2 weeks each)

### 11. Find Dermatologists (Location-Based)
- Geolocation API for user location
- Search dermatologists via external API
- Display results on map
- Filter by insurance, ratings, distance
- **Tech:** Geolocation API, Google Maps/Mapbox, dermatologist directory API
- **Files:** `src/components/dermatologist/FindProvider.tsx`

### 12. PWA Offline Support
- Service worker for offline functionality
- Cache API for storing images/results
- Offline queue for analyses
- **Tech:** Workbox, Service Workers
- **Files:** `public/sw.js`, `vite.config.ts`

### 13. Camera Guidance Overlay
- Real-time camera preview
- Guide user to proper distance/lighting
- Brightness/blur detection
- **Tech:** getUserMedia, canvas analysis
- **Files:** `src/components/upload/GuidedCamera.tsx`

### 14. Analysis History with Charts
- Visualize risk score trends over time
- Line/bar charts for multiple analyses
- Export chart as image
- **Tech:** Recharts or Chart.js
- **Files:** `src/components/history/TrendChart.tsx`

### 15. Email Results
- Email input form
- Send results via backend endpoint
- Email template with HTML formatting
- **Tech:** Backend endpoint, email service (SendGrid/Resend)
- **Files:** `app/routes/email.py`, `src/components/results/EmailForm.tsx`

## Phase 4: Complex Integrations (2-4 weeks each)

### 16. User Accounts & Authentication
- Sign up/login flow
- Email verification
- Password reset
- Session management
- **Tech:** Auth provider (Clerk, Auth0, or custom JWT)
- **Files:** `src/contexts/AuthContext.tsx`, backend auth routes

### 17. Cloud Storage for Analysis History
- Store images and results in cloud
- Sync across devices
- Privacy/encryption
- **Tech:** Cloud storage (S3, Cloudflare R2), database (Supabase, Firebase)
- **Files:** Backend storage service, API routes

### 18. Appointment Scheduling Integration
- Calendar view of available slots
- Book appointments with dermatologists
- Email/SMS confirmations
- **Tech:** Calendly API, Acuity Scheduling, or custom system
- **Files:** `src/components/scheduling/AppointmentBooking.tsx`

### 19. Health App Integration
- Export to Apple Health (iOS)
- Export to Google Fit (Android)
- **Tech:** HealthKit (iOS), Health Connect (Android), Capacitor
- **Files:** Mobile app wrappers, native plugins

### 20. Telemedicine Consultation
- Video call integration
- Screen sharing (show results to doctor)
- Chat/messaging
- Payment processing
- **Tech:** Twilio Video, Agora, Daily.co
- **Files:** `src/components/telemedicine/VideoCall.tsx`

## Phase 5: Research & Medical Features (1-3 months each)

### 21. Multi-Model Ensemble Analysis
- Run analysis through multiple AI models
- Aggregate predictions
- Show consensus/disagreement
- **Tech:** Multiple model endpoints, aggregation logic
- **Files:** Backend model ensemble, UI confidence visualization

### 22. Dermatologist Review Queue
- Submit cases for human review
- Real dermatologist second opinion
- Flag disagreements between AI and doctor
- **Tech:** Admin dashboard, notification system, HIPAA compliance
- **Files:** Admin interface, review workflow

### 23. Clinical Trial Matching
- Match users with relevant skin cancer trials
- Integration with ClinicalTrials.gov API
- Eligibility screening
- **Tech:** ClinicalTrials.gov API, matching algorithm
- **Files:** `src/components/trials/TrialMatcher.tsx`

### 24. FHIR Medical Records Integration
- Import patient history
- Export results to EHR systems
- HL7 FHIR compliance
- **Tech:** FHIR API, healthcare interoperability standards
- **Files:** Backend FHIR integration, security compliance

### 25. Insurance Pre-Authorization
- Check coverage for dermatology visits
- Submit pre-auth requests
- Integration with insurance APIs
- **Tech:** Insurance eligibility APIs (Change Healthcare, Availity)
- **Files:** Backend insurance integration

## Implementation Priority

**Immediate (Ship Next):**
1. Download Results as Image
2. Copy Results to Clipboard
3. Tutorial/Onboarding Modal

**Short Term (Next Month):**
4. Save Analysis History (Local)
5. Dark Mode Toggle
6. Export Results as PDF

**Medium Term (Next Quarter):**
7. Find Dermatologists
8. PWA Offline Support
9. Email Results

**Long Term (Future Roadmap):**
10. User Accounts & Cloud Storage
11. Appointment Scheduling
12. Telemedicine Consultation
