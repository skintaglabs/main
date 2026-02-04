const urlParams = new URLSearchParams(window.location.search)

export const API_URL = urlParams.get('api') ||
                       (import.meta as any).env?.VITE_API_URL ||
                       'http://localhost:8000'
