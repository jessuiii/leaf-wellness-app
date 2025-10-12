// Configure your Python backend API URL here
export const API_CONFIG = {
  PYTHON_BACKEND_URL: import.meta.env.VITE_PYTHON_API_URL || 'http://localhost:5000',
  ANALYZE_ENDPOINT: '/predict'
};
