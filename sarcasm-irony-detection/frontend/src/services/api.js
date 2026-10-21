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

export const DEFAULT_MODELS = [
  {
    id: 'bow_linearsvm',
    name: 'LinearSVM (BoW)',
    full_name: 'Linear Support Vector Classifier with Bag of Words (One-vs-Rest)',
    paradigm: 'Bag of Words (n-grams 1-2)',
    parameter_count: 112312,
    val_macro_f1: 0.7187,
    val_accuracy: 0.8457,
    latency_ms: 0.027,
    is_champion: true,
    description: 'Selected champion model. Achieves top Macro F1 (0.7187), avoids minority class collapse, and executes ultra-low latency inference.',
  },
  {
    id: 'tfidf_linearsvm',
    name: 'LinearSVM (TF-IDF)',
    full_name: 'Linear Support Vector Classifier with Sublinear TF-IDF',
    paradigm: 'TF-IDF + Statistical N-grams',
    parameter_count: 112312,
    val_macro_f1: 0.7002,
    val_accuracy: 0.8565,
    latency_ms: 0.014,
    is_champion: false,
    description: 'High accuracy classical model leveraging inverse document frequency weighting across 28,077 unigram/bigram tokens.',
  },
  {
    id: 'glove_bilstm',
    name: 'GloVe + Bi-LSTM',
    full_name: 'Bidirectional LSTM with 100d Pretrained GloVe Embeddings (BILSTM-02)',
    paradigm: 'Dense Word Embeddings (Deep Learning)',
    parameter_count: 3197636,
    val_macro_f1: 0.6940,
    val_accuracy: 0.8612,
    latency_ms: 0.041,
    is_champion: false,
    description: 'Recurrent neural sequence model capturing bidirectional token contexts using 100-dimensional GloVe word vectors.',
  },
  {
    id: 'fasttext_attention',
    name: 'FastText + Bi-LSTM + Attention',
    full_name: 'FastText 300d + Bi-LSTM + Feed-Forward Attention Mechanism (FT-02)',
    paradigm: 'Subword Embeddings & Feed-Forward Attention',
    parameter_count: 9204828,
    val_macro_f1: 0.6928,
    val_accuracy: 0.8603,
    latency_ms: 0.098,
    is_champion: false,
    description: 'Resolves social media OOV slang via character n-grams and weights salient contrastive tokens through additive attention.',
  },
  {
    id: 'bert_transformer',
    name: 'BERT Contextual',
    full_name: 'bert-base-uncased (12-layer Bidirectional Transformer)',
    paradigm: 'Pretrained Contextual Transformer (WordPiece)',
    parameter_count: 109485316,
    val_macro_f1: 0.6935,
    val_accuracy: 0.8612,
    latency_ms: 39.33,
    is_champion: false,
    description: 'Deep contextual transformer fine-tuned end-to-end; completely eliminates mutual confusion between Sarcasm and Irony.',
  },
];

/**
 * Predict Sarcasm/Irony for a single input text using chosen model.
 * @param {string} text - Raw input text (max 5000 chars)
 * @param {string} [model='bow_linearsvm'] - Chosen model ID
 * @param {AbortSignal} [signal] - Optional AbortController signal
 * @returns {Promise<Object>} Formatted prediction object
 */
export async function predictText(text, model = 'bow_linearsvm', signal) {
  try {
    const payload = { text, model };
    const response = await apiClient.post('/api/predict', payload, { signal });
    if (response.data && response.data.success && response.data.data) {
      return response.data.data;
    }
    throw new Error('Malformed prediction response from backend.');
  } catch (error) {
    throw normalizeError(error);
  }
}

/**
 * Fetch all available comparative NLP models from the backend / ML service.
 * @param {AbortSignal} [signal]
 * @returns {Promise<Array<Object>>}
 */
export async function getAvailableModels(signal) {
  try {
    const response = await apiClient.get('/api/models', { signal });
    if (response.data && response.data.data && Array.isArray(response.data.data.models)) {
      return response.data.data.models;
    }
    return DEFAULT_MODELS;
  } catch {
    return DEFAULT_MODELS;
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
      // Static verified baseline fallback if network is completely down
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
  getAvailableModels,
  DEFAULT_MODELS,
  getPredictionHistory,
  deletePrediction,
  checkHealth,
  getModelInfo,
};

