/**
 * Prediction Mongoose Model.
 * Represents stored prediction history with validation and traceability.
 */

const mongoose = require('mongoose');

const ProbabilitySchema = new mongoose.Schema(
  {
    Regular: {
      type: Number,
      required: true,
      min: [0, 'Probability cannot be less than 0'],
      max: [1, 'Probability cannot be greater than 1'],
    },
    Sarcasm: {
      type: Number,
      required: true,
      min: [0, 'Probability cannot be less than 0'],
      max: [1, 'Probability cannot be greater than 1'],
    },
    Irony: {
      type: Number,
      required: true,
      min: [0, 'Probability cannot be less than 0'],
      max: [1, 'Probability cannot be greater than 1'],
    },
    Figurative: {
      type: Number,
      required: true,
      min: [0, 'Probability cannot be less than 0'],
      max: [1, 'Probability cannot be greater than 1'],
    },
  },
  { _id: false }
);

const PredictionSchema = new mongoose.Schema(
  {
    text: {
      type: String,
      required: [true, 'Input text is required'],
      trim: true,
      maxlength: [5000, 'Text length cannot exceed 5000 characters'],
    },
    predictedClass: {
      type: String,
      required: [true, 'Predicted class is required'],
      enum: {
        values: ['Regular', 'Sarcasm', 'Irony', 'Figurative'],
        message: '{VALUE} is not a valid prediction class',
      },
    },
    confidence: {
      type: Number,
      required: [true, 'Prediction confidence score is required'],
      min: [0, 'Confidence must be at least 0'],
      max: [1, 'Confidence cannot exceed 1'],
    },
    probabilities: {
      type: ProbabilitySchema,
      required: [true, 'Probability distribution is required'],
    },
    modelName: {
      type: String,
      required: [true, 'Model name is required for traceability'],
      trim: true,
    },
    modelVersion: {
      type: String,
      required: [true, 'Model version is required for auditability'],
      trim: true,
      index: true,
    },
    latencyMs: {
      type: Number,
      min: 0,
      default: null,
    },
    createdAt: {
      type: Date,
      default: Date.now,
      index: true,
    },
  },
  {
    timestamps: true,
    toJSON: {
      virtuals: true,
      transform: (doc, ret) => {
        ret.id = ret._id;
        delete ret._id;
        delete ret.__v;
        return ret;
      },
    },
  }
);

// Compound indexes for history querying
PredictionSchema.index({ createdAt: -1 });
PredictionSchema.index({ predictedClass: 1, createdAt: -1 });

module.exports = mongoose.model('Prediction', PredictionSchema);
