/**
 * ML Service Client.
 * Manages HTTP communication between Node/Express backend and Python FastAPI ML service.
 * Handles timeouts, network errors, contract validation, and error normalization.
 */

const axios = require('axios');
const config = require('../config/env');

class MLServiceError extends Error {
  constructor(message, code, statusCode = 502, details = null) {
    super(message);
    this.name = 'MLServiceError';
    this.code = code;
    this.statusCode = statusCode;
    this.details = details;
  }
}

class MLService {
  constructor() {
    this.client = axios.create({
      baseURL: config.mlServiceUrl,
      timeout: config.mlServiceTimeoutMs,
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
    });
  }

  /**
   * Check ML service health status.
   * @returns {Promise<Object>} Health check payload
   */
  async checkHealth() {
    try {
      const response = await this.client.get('/health');
      return response.data;
    } catch (error) {
      return {
        status: 'error',
        service: 'ml-service',
        model_loaded: false,
        error: error.message,
      };
    }
  }

  /**
   * Check ML service readiness probe.
   * @returns {Promise<boolean>}
   */
  async isReady() {
    try {
      const response = await this.client.get('/ready');
      return response.status === 200 && response.data.ready === true;
    } catch (error) {
      return false;
    }
  }

  /**
   * Fetch model metadata.
   * @returns {Promise<Object>}
   */
  async getModelInfo() {
    try {
      const response = await this.client.get('/model-info');
      return response.data;
    } catch (error) {
      this._handleAxiosError(error, 'Failed to fetch model info from ML service.');
    }
  }

  /**
   * Fetch all 5 available comparative models from ML service.
   * @returns {Promise<Object>} Available models payload
   */
  async getAvailableModels() {
    try {
      const response = await this.client.get('/models');
      return response.data;
    } catch (error) {
      this._handleAxiosError(error, 'Failed to fetch available models from ML service.');
    }
  }

  /**
   * Request single text prediction from the ML service.
   * @param {string} text - Raw input text to classify
   * @param {string} [model] - Target model ID (e.g. bow_linearsvm, bert_transformer)
   * @returns {Promise<Object>} Formatted prediction result
   */
  async predict(text, model = null) {
    try {
      const payload = { text };
      if (model) {
        payload.model = model;
      }
      const response = await this.client.post('/predict', payload);
      const data = response.data;

      // Contract validation (Section 50 & Section 74)
      this._validatePredictionContract(data);

      return {
        text: data.text,
        predictedClass: data.predicted_class,
        confidence: Number(data.confidence),
        probabilities: {
          Regular: Number(data.probabilities.Regular),
          Sarcasm: Number(data.probabilities.Sarcasm),
          Irony: Number(data.probabilities.Irony),
          Figurative: Number(data.probabilities.Figurative),
        },
        modelName: data.model_name,
        modelVersion: data.model_version,
        latencyMs: data.latency_ms !== undefined ? Number(data.latency_ms) : null,
      };
    } catch (error) {
      this._handleAxiosError(error, 'ML service prediction failed.');
    }
  }

  /**
   * Request batch prediction from the ML service.
   * @param {Array<string>} texts - Array of raw input texts
   * @param {string} [model] - Target model ID
   * @returns {Promise<Object>}
   */
  async predictBatch(texts, model = null) {
    try {
      const payload = { texts };
      if (model) {
        payload.model = model;
      }
      const response = await this.client.post('/predict/batch', payload);
      const data = response.data;

      if (!Array.isArray(data.predictions)) {
        throw new MLServiceError(
          'Invalid batch response from ML service.',
          'INVALID_UPSTREAM_RESPONSE',
          502
        );
      }

      const formatted = data.predictions.map((p) => {
        this._validatePredictionContract(p);
        return {
          text: p.text,
          predictedClass: p.predicted_class,
          confidence: Number(p.confidence),
          probabilities: {
            Regular: Number(p.probabilities.Regular),
            Sarcasm: Number(p.probabilities.Sarcasm),
            Irony: Number(p.probabilities.Irony),
            Figurative: Number(p.probabilities.Figurative),
          },
          modelName: p.model_name,
          modelVersion: p.model_version,
          latencyMs: p.latency_ms !== undefined ? Number(p.latency_ms) : null,
        };
      });

      return {
        predictions: formatted,
        totalCount: data.total_count,
        batchLatencyMs: data.batch_latency_ms,
        modelName: data.model_name,
        modelVersion: data.model_version,
      };
    } catch (error) {
      this._handleAxiosError(error, 'ML service batch prediction failed.');
    }
  }

  /**
   * Validates that the payload from the ML service adheres to the required contract.
   * @param {Object} data - Prediction payload from Python
   */
  _validatePredictionContract(data) {
    const validClasses = ['Regular', 'Sarcasm', 'Irony', 'Figurative'];

    if (!data || typeof data !== 'object') {
      throw new MLServiceError(
        'Malformed response payload from ML service.',
        'INVALID_UPSTREAM_RESPONSE',
        502
      );
    }

    if (!data.predicted_class || !validClasses.includes(data.predicted_class)) {
      throw new MLServiceError(
        `Invalid or unrecognized predicted class: "${data.predicted_class}"`,
        'INVALID_UPSTREAM_RESPONSE',
        502
      );
    }

    if (typeof data.confidence !== 'number' || data.confidence < 0 || data.confidence > 1) {
      throw new MLServiceError(
        'Invalid confidence score from ML service.',
        'INVALID_UPSTREAM_RESPONSE',
        502
      );
    }

    if (!data.probabilities || typeof data.probabilities !== 'object') {
      throw new MLServiceError(
        'Missing probability distribution from ML service.',
        'INVALID_UPSTREAM_RESPONSE',
        502
      );
    }

    for (const cls of validClasses) {
      if (typeof data.probabilities[cls] !== 'number') {
        throw new MLServiceError(
          `Missing or non-numeric probability for class: ${cls}`,
          'INVALID_UPSTREAM_RESPONSE',
          502
        );
      }
    }
  }

  /**
   * Normalizes Axios errors into standard MLServiceError instances.
   */
  _handleAxiosError(error, defaultMessage) {
    if (error instanceof MLServiceError) {
      throw error;
    }

    // Connection timeout
    if (error.code === 'ECONNABORTED' || (error.message && error.message.includes('timeout'))) {
      throw new MLServiceError(
        `ML inference request timed out after ${config.mlServiceTimeoutMs}ms.`,
        'ML_SERVICE_TIMEOUT',
        504
      );
    }

    // Connection refused / Network down
    if (error.code === 'ECONNREFUSED' || error.code === 'ENOTFOUND') {
      throw new MLServiceError(
        'ML inference service is currently unreachable or stopped.',
        'ML_SERVICE_UNAVAILABLE',
        503
      );
    }

    // Upstream HTTP response errors
    if (error.response) {
      const status = error.response.status;
      const upstreamError = error.response.data || {};
      const message = upstreamError.message || upstreamError.error || defaultMessage;

      if (status === 422) {
        throw new MLServiceError(message, 'VALIDATION_ERROR', 422, upstreamError);
      }
      if (status === 503) {
        throw new MLServiceError(
          'ML inference model is currently not loaded or unavailable.',
          'ML_SERVICE_UNAVAILABLE',
          503,
          upstreamError
        );
      }
      throw new MLServiceError(message, 'ML_PREDICTION_FAILED', 502, upstreamError);
    }

    // Other unexpected errors
    throw new MLServiceError(
      `${defaultMessage} ${error.message}`,
      'ML_SERVICE_ERROR',
      502
    );
  }
}

module.exports = new MLService();
module.exports.MLService = MLService;
module.exports.MLServiceError = MLServiceError;
