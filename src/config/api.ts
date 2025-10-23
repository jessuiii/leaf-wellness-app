// Configure your Python backend API URL here
export const API_CONFIG = {
  PYTHON_BACKEND_URL: import.meta.env.VITE_PYTHON_API_URL || 'http://localhost:5000',
  ANALYZE_ENDPOINT: '/predict',
  PREDICT_UPLOAD_ENDPOINT: '/predict-upload',
  PLANT_HISTORY_ENDPOINT: '/plant/{plantId}/history',
  PLANT_ENDPOINT: '/plant/{plantId}',
  CREATE_PLANT_ENDPOINT: '/plant/{plantId}/create',
  ALL_PLANTS_ENDPOINT: '/plants',
  TREATMENT_ENDPOINT: '/plant/{plantId}/treatment',
  STATUS_SUMMARY_ENDPOINT: '/plants/status-summary'
};

// API helper functions
export const apiEndpoints = {
  predict: () => `${API_CONFIG.PYTHON_BACKEND_URL}${API_CONFIG.ANALYZE_ENDPOINT}`,
  predictUpload: () => `${API_CONFIG.PYTHON_BACKEND_URL}${API_CONFIG.PREDICT_UPLOAD_ENDPOINT}`,
  plantHistory: (plantId: string) => 
    `${API_CONFIG.PYTHON_BACKEND_URL}${API_CONFIG.PLANT_HISTORY_ENDPOINT.replace('{plantId}', plantId)}`,
  plant: (plantId: string) => 
    `${API_CONFIG.PYTHON_BACKEND_URL}${API_CONFIG.PLANT_ENDPOINT.replace('{plantId}', plantId)}`,
  createPlant: (plantId: string) => 
    `${API_CONFIG.PYTHON_BACKEND_URL}${API_CONFIG.CREATE_PLANT_ENDPOINT.replace('{plantId}', plantId)}`,
  allPlants: () => `${API_CONFIG.PYTHON_BACKEND_URL}${API_CONFIG.ALL_PLANTS_ENDPOINT}`,
  addTreatment: (plantId: string) => 
    `${API_CONFIG.PYTHON_BACKEND_URL}${API_CONFIG.TREATMENT_ENDPOINT.replace('{plantId}', plantId)}`,
  statusSummary: () => `${API_CONFIG.PYTHON_BACKEND_URL}${API_CONFIG.STATUS_SUMMARY_ENDPOINT}`
};
