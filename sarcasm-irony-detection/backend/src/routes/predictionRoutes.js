/**
 * Prediction Routes.
 * Mounts prediction and prediction-history endpoints.
 */

const express = require('express');
const {
  predict,
  predictBatch,
  getPredictions,
  deletePrediction,
} = require('../controllers/predictionController');
const {
  validatePredictRequest,
  validateBatchPredictRequest,
  validateHistoryQuery,
} = require('../middleware/validation');

const router = express.Router();

router.post('/predict', validatePredictRequest, predict);
router.post('/predict/batch', validateBatchPredictRequest, predictBatch);
router.get('/predictions', validateHistoryQuery, getPredictions);
router.delete('/predictions/:id', deletePrediction);

module.exports = router;
