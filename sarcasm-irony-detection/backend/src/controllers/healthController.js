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

module.exports = {
  getHealth,
};
