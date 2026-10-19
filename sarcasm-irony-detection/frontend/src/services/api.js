/**
 * Centralized API Client Service.
 * Communicates ONLY with the Node.js + Express Application Backend.
 * Does NOT communicate directly with Python ML service (port 8000) or MongoDB.
 */

import axios from 'axios';

// Node.js Express backend base URL from Vite environment variable
const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000';

const apiClient = axios.create({
  baseURL: BASE_URL,
  timeout: 10000, // 10 second timeout
  headers: {
    'Content-Type': 'application/json',
    Accept: 'application/json',
  },
});

/**
 * Normalize and extract user-friendly error messages from API failures.
 * Never exposes raw stack traces, database strings, or internal paths.
 */
function normalizeError(error) {
  if (axios.isCancel(error) || error.name === 'CanceledError') {
    return {
      isCanceled: true,
      message: 'Request was canceled.',
    };
  }

  // Network offline / connection refused
  if (error.code === 'ECONNREFUSED' || error.code === 'ERR_NETWORK' || !error.response) {
    return {
      code: 'NETWORK_ERROR',
      message: 'Prediction service is currently unreachable. Please ensure the backend server is running.',
      statusCode: 503,
    };
  }

  const { status, data } = error.response;
  const serverError = data && data.error ? data.error : null;

  if (status === 400) {
    return {
      code: serverError?.code || 'VALIDATION_ERROR',
      message: serverError?.message || 'Invalid input. Please check the entered text and try again.',
      statusCode: 400,
    };
  }

  if (status === 429) {
    return {
      code: 'RATE_LIMIT_EXCEEDED',
      message: 'Too many requests. Please wait a moment before analyzing more text.',
      statusCode: 429,
    };
  }

  if (status === 503) {
    return {
      code: 'SERVICE_UNAVAILABLE',
      message: 'The ML inference service is temporarily unavailable. Please retry shortly.',
      statusCode: 503,
    };
  }

  if (status === 504) {
    return {
      code: 'TIMEOUT',
      message: 'The analysis timed out. The server might be experiencing high load.',
      statusCode: 504,
    };
  }

  return {
    code: 'UNKNOWN_ERROR',
    message: serverError?.message || 'Unable to complete request right now. Please try again.',
    statusCode: status || 500,
  };
}

/**
 * Predict Sarcasm/Irony for a single input text.
 * @param {string} text - Raw input text (max 5000 chars)
 * @param {AbortSignal} [signal] - Optional AbortController signal
 * @returns {Promise<Object>} Formatted prediction object
 */
export async function predictText(text, signal) {
  try {
    const response = await apiClient.post('/api/predict', { text }, { signal });
    if (response.data && response.data.success && response.data.data) {
      return response.data.data;
    }
    throw new Error('Malformed prediction response from backend.');
  } catch (error) {
    throw normalizeError(error);
  }
}

/**
 * Retrieve prediction history from the backend.
 * @param {Object} options - { limit, page, predictedClass }
 * @param {AbortSignal} [signal]
 */
export async function getPredictionHistory({ limit = 20, page = 1, predictedClass = '' } = {}, signal) {
  try {
    const params = { limit, page };
    if (predictedClass) {
      params.predictedClass = predictedClass;
    }

    const response = await apiClient.get('/api/predictions', { params, signal });
    return {
      predictions: response.data.data || [],
      pagination: response.data.pagination || { total: 0, page: 1, limit: 20, pages: 1 },
      message: response.data.message || null,
    };
  } catch (error) {
    throw normalizeError(error);
  }
}

/**
 * Delete a specific prediction record from history.
 * @param {string} id - Prediction ObjectId hex string
 */
export async function deletePrediction(id) {
  try {
    const response = await apiClient.delete(`/api/predictions/${encodeURIComponent(id)}`);
    return response.data;
  } catch (error) {
    throw normalizeError(error);
  }
}

/**
 * Check health status of backend and upstream services.
 */
export async function checkHealth(signal) {
  try {
    const response = await apiClient.get('/api/health', { signal });
    return response.data;
  } catch (error) {
    return {
      success: false,
      service: 'backend',
      status: 'offline',
      components: {
        mlService: { status: 'unavailable', modelLoaded: false },
        database: { status: 'disconnected' },
      },
      timestamp: new Date().toISOString(),
    };
  }
}

/**
 * Retrieve Model Metadata & Performance Metrics.
 */
export async function getModelInfo(signal) {
  try {
    // Attempt /api/model-info first, then fallback to /model-info
    const response = await apiClient.get('/api/model-info', { signal });
    return response.data.data;
  } catch {
    try {
      const fallback = await apiClient.get('/model-info', { signal });
      return fallback.data.data;
    } catch {
      // Static verified Phase 9 baseline fallback if network is completely down
      return {
        model_name: 'SarcasmIronyClassifier-LinearSVM',
        model_version: 'SIC-v1.0',
        model_type: 'LinearSVC (One-vs-Rest) with BoW CountVectorizer',
        classes: ['Regular', 'Sarcasm', 'Irony', 'Figurative'],
        macro_f1: 0.7719,
        accuracy: 0.77,
        status: 'client_cached',
      };
    }
  }
}

export default {
  predictText,
  getPredictionHistory,
  deletePrediction,
  checkHealth,
  getModelInfo,
};
