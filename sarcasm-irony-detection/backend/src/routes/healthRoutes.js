/**
 * Health check routes.
 * GET /api/health
 */

const express = require('express');
const { getHealth, getModelInfo } = require('../controllers/healthController');

const router = express.Router();

router.get('/health', getHealth);
router.get('/model-info', getModelInfo);

module.exports = router;

