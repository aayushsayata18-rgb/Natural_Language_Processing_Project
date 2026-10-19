/**
 * Health Controller.
 * Aggregates health status of Node backend, MongoDB, and Python ML service.
 */

const mlService = require('../services/mlService');
const { isDbConnected } = require('../config/database');

/**
 * Health check endpoint for the Express backend.
 * GET /api/health
 */
const getHealth = async (req, res) => {
  const mlHealth = await mlService.checkHealth();
  const dbConnected = isDbConnected();

  const isMlAvailable = mlHealth.status === 'ok' && mlHealth.model_loaded === true;

  const responsePayload = {
    success: true,
    service: 'backend',
    status: isMlAvailable ? 'healthy' : 'degraded',
    components: {
      mlService: {
        status: isMlAvailable ? 'available' : 'unavailable',
        modelLoaded: Boolean(mlHealth.model_loaded),
        modelName: mlHealth.model_name || null,
        modelVersion: mlHealth.model_version || null,
      },
      database: {
        status: dbConnected ? 'connected' : 'disconnected',
      },
    },
    timestamp: new Date().toISOString(),
  };

  // Status code: 200 even if degraded, but if ML is down indicates degraded status
  return res.status(200).json(responsePayload);
};

/**
 * Get Model Information metadata.
 * GET /api/model-info or GET /model-info
 */
const getModelInfo = async (req, res) => {
  try {
    const mlInfo = await mlService.getModelInfo();
    return res.status(200).json({
      success: true,
      data: mlInfo,
    });
  } catch (error) {
    // Graceful fallback to verified Phase 9/10 packaged model specifications
    return res.status(200).json({
      success: true,
      data: {
        model_name: 'SarcasmIronyClassifier-LinearSVM',
        model_version: 'SIC-v1.0',
        model_type: 'LinearSVC (One-vs-Rest) with BoW CountVectorizer',
        classes: ['Regular', 'Sarcasm', 'Irony', 'Figurative'],
        macro_f1: 0.7719,
        accuracy: 0.7700,
        dataset: 'SemEval-2018 Task 3 Subtask B',
        status: 'cached_fallback',
      },
    });
  }
};

module.exports = {
  getHealth,
  getModelInfo,
};

