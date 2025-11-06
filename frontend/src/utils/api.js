/**
 * API Client for backend communication
 */

import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json'
  }
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add any auth tokens here if needed
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// ============================================================
// OPTIMIZATION ENDPOINTS
// ============================================================

export const optimizationAPI = {
  start: (config) => api.post('/optimization/start', config),
  getStatus: (runId) => api.get(`/optimization/${runId}/status`),
  pause: (runId) => api.post(`/optimization/${runId}/pause`),
  resume: (runId) => api.post(`/optimization/${runId}/resume`),
  stop: (runId) => api.post(`/optimization/${runId}/stop`),
  list: () => api.get('/optimization/list')
};

// ============================================================
// TELEMETRY ENDPOINTS
// ============================================================

export const telemetryAPI = {
  getTrials: (runId) => api.get(`/telemetry/${runId}/trials`),
  getTrialData: (runId, trialId) => api.get(`/telemetry/${runId}/trial/${trialId}`)
};

// ============================================================
// PARAMETERS ENDPOINTS
// ============================================================

export const parametersAPI = {
  getBounds: () => api.get('/parameters/bounds'),
  getDefaults: () => api.get('/parameters/defaults'),
  export: (runId, format = 'parm') => api.post(`/parameters/export/${runId}`, null, {
    params: { format },
    responseType: 'blob'
  })
};

// ============================================================
// ANALYSIS ENDPOINTS
// ============================================================

export const analysisAPI = {
  getCorrelation: (runId) => api.get(`/analysis/${runId}/correlation`),
  getFrequencyResponse: (runId, axis = 'roll') => api.get(`/analysis/${runId}/frequency_response`, {
    params: { axis }
  })
};

// ============================================================
// DRONE SPECS
// ============================================================

export const droneAPI = {
  getSpecs: () => api.get('/drone/specs')
};

// ============================================================
// HEALTH CHECK
// ============================================================

export const healthAPI = {
  check: () => api.get('/health')
};

export default api;
