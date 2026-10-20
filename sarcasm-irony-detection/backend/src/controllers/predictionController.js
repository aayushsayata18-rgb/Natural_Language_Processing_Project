/**
 * Prediction Controller.
 * Coordinates input validation, ML service prediction, and MongoDB history persistence.
 */

const mongoose = require('mongoose');
const Prediction = require('../models/Prediction');
const mlService = require('../services/mlService');
const { isDbConnected } = require('../config/database');

/**
 * Handle single text prediction request.
 * POST /api/predict
 */
const predict = async (req, res, next) => {
  try {
    const { text, model } = req.body;

    // Call Python ML service with selected model
    const mlResult = await mlService.predict(text, model);

    let savedRecord = null;
    let historySaved = false;

    // Conditionally save to MongoDB if connected (Graceful degradation per Section 57)
    if (isDbConnected()) {
      try {
        savedRecord = await Prediction.create({
          text: mlResult.text,
          predictedClass: mlResult.predictedClass,
          confidence: mlResult.confidence,
          probabilities: mlResult.probabilities,
          modelName: mlResult.modelName,
          modelVersion: mlResult.modelVersion,
          latencyMs: mlResult.latencyMs,
        });
        historySaved = true;
      } catch (dbError) {
        console.warn('[PredictionController] Failed to persist history to MongoDB:', dbError.message);
        historySaved = false;
      }
    }

    // Standardized response matching Section 28 & Section 88 contract
    return res.status(200).json({
      success: true,
      data: {
        text: mlResult.text,
        predicted_class: mlResult.predictedClass,
        predictedClass: mlResult.predictedClass,
        confidence: mlResult.confidence,
        probabilities: mlResult.probabilities,
        model_name: mlResult.modelName,
        modelName: mlResult.modelName,
        model_version: mlResult.modelVersion,
        modelVersion: mlResult.modelVersion,
        latency_ms: mlResult.latencyMs,
        history_saved: historySaved,
        id: savedRecord ? savedRecord.id : null,
      },
    });
  } catch (error) {
    next(error);
  }
};

/**
 * Handle batch prediction request.
 * POST /api/predict/batch
 */
const predictBatch = async (req, res, next) => {
  try {
    const { texts, model } = req.body;

    const batchResult = await mlService.predictBatch(texts, model);

    // Save batch to MongoDB in bulk if connected
    if (isDbConnected()) {
      try {
        const docs = batchResult.predictions.map((p) => ({
          text: p.text,
          predictedClass: p.predictedClass,
          confidence: p.confidence,
          probabilities: p.probabilities,
          modelName: p.modelName,
          modelVersion: p.modelVersion,
          latencyMs: p.latencyMs,
        }));
        await Prediction.insertMany(docs, { ordered: false });
      } catch (dbError) {
        console.warn('[PredictionController] Failed to batch persist history:', dbError.message);
      }
    }

    return res.status(200).json({
      success: true,
      data: {
        predictions: batchResult.predictions.map((p) => ({
          text: p.text,
          predicted_class: p.predictedClass,
          predictedClass: p.predictedClass,
          confidence: p.confidence,
          probabilities: p.probabilities,
          model_name: p.modelName,
          model_version: p.modelVersion,
          latency_ms: p.latencyMs,
        })),
        total_count: batchResult.totalCount,
        batch_latency_ms: batchResult.batchLatencyMs,
        model_name: batchResult.modelName,
        model_version: batchResult.modelVersion,
      },
    });
  } catch (error) {
    next(error);
  }
};

/**
 * Retrieve recent prediction history.
 * GET /api/predictions?limit=20&page=1
 */
const getPredictions = async (req, res, next) => {
  try {
    if (!isDbConnected()) {
      return res.status(200).json({
        success: true,
        data: [],
        pagination: {
          total: 0,
          limit: req.query.limit,
          page: req.query.page,
        },
        message: 'Database is currently offline; history is unavailable.',
      });
    }

    const { limit, page, predictedClass } = req.query;
    const filter = {};
    if (predictedClass) {
      filter.predictedClass = predictedClass;
    }

    const skip = (page - 1) * limit;

    const [records, total] = await Promise.all([
      Prediction.find(filter)
        .sort({ createdAt: -1 })
        .skip(skip)
        .limit(limit)
        .lean(),
      Prediction.countDocuments(filter),
    ]);

    const formatted = records.map((r) => ({
      id: r._id.toString(),
      text: r.text,
      predicted_class: r.predictedClass,
      predictedClass: r.predictedClass,
      confidence: r.confidence,
      probabilities: r.probabilities,
      model_name: r.modelName,
      model_version: r.modelVersion,
      modelVersion: r.modelVersion,
      created_at: r.createdAt,
      createdAt: r.createdAt,
    }));

    return res.status(200).json({
      success: true,
      data: formatted,
      pagination: {
        total,
        page,
        limit,
        pages: Math.ceil(total / limit) || 1,
      },
    });
  } catch (error) {
    next(error);
  }
};

/**
 * Delete a specific prediction record by ID.
 * DELETE /api/predictions/:id
 */
const deletePrediction = async (req, res, next) => {
  try {
    const { id } = req.params;

    if (!mongoose.Types.ObjectId.isValid(id)) {
      return res.status(400).json({
        success: false,
        error: {
          code: 'INVALID_IDENTIFIER',
          message: `The provided identifier "${id}" is not a valid ObjectId.`,
        },
      });
    }

    if (!isDbConnected()) {
      return res.status(503).json({
        success: false,
        error: {
          code: 'DATABASE_UNAVAILABLE',
          message: 'Database is currently offline; cannot delete history records.',
        },
      });
    }

    const deleted = await Prediction.findByIdAndDelete(id);
    if (!deleted) {
      return res.status(404).json({
        success: false,
        error: {
          code: 'PREDICTION_NOT_FOUND',
          message: `Prediction with id "${id}" was not found.`,
        },
      });
    }

    return res.status(200).json({
      success: true,
      message: 'Prediction history record deleted successfully.',
      data: { id },
    });
  } catch (error) {
    next(error);
  }
};

/**
 * Handle get available comparative models request.
 * GET /api/models
 */
const getModels = async (req, res, next) => {
  try {
    const modelsData = await mlService.getAvailableModels();
    return res.status(200).json({
      success: true,
      data: modelsData,
    });
  } catch (error) {
    next(error);
  }
};

module.exports = {
  predict,
  predictBatch,
  getPredictions,
  deletePrediction,
  getModels,
};

