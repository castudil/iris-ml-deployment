// API configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000/api';

export const API_ENDPOINTS = {
  health: `${API_BASE_URL}/health/`,
  models: `${API_BASE_URL}/models/`,
  predict: `${API_BASE_URL}/predict/`,
};

export default API_BASE_URL;
