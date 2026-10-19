/**
 * Centralized Error Handling Middleware.
 * Catches and formats all unhandled errors into standardized JSON responses.
 * Never leaks internal file paths, credentials, or stack traces.
 */

const { MLServiceError } = require('../services/mlService');

// eslint-disable-next-line no-unused-vars
const errorHandler = (err, req, res, next) => {
  // Log server errors for internal monitoring
  if (process.env.NODE_ENV !== 'test') {
    console.error(`[Error] ${req.method} ${req.originalUrl}:`, err.message);
  }

  // Handle JSON body parser syntax errors
  if (err instanceof SyntaxError && err.status === 400 && 'body' in err) {
    return res.status(400).json({
      success: false,
      error: {
        code: 'INVALID_JSON',
        message: 'Request body contains malformed JSON syntax.',
      },
    });
  }

  // Handle ML Service Client Errors
  if (err instanceof MLServiceError || err.name === 'MLServiceError' || (err.statusCode && err.code && err.code.startsWith('ML_'))) {
    return res.status(err.statusCode || 502).json({
      success: false,
      error: {
        code: err.code,
        message: err.message,
      },
    });
  }

  // Handle Mongoose Validation Errors
  if (err.name === 'ValidationError') {
    const messages = Object.values(err.errors).map((val) => val.message);
    return res.status(400).json({
      success: false,
      error: {
        code: 'VALIDATION_ERROR',
        message: messages.join(', '),
      },
    });
  }

  // Handle Mongoose Cast Errors (e.g. invalid ObjectId)
  if (err.name === 'CastError') {
    return res.status(400).json({
      success: false,
      error: {
        code: 'INVALID_IDENTIFIER',
        message: `Resource with identifier "${err.value}" is invalid.`,
      },
    });
  }

  // Default catch-all internal server error
  const statusCode = err.statusCode || 500;
  return res.status(statusCode).json({
    success: false,
    error: {
      code: err.code || 'INTERNAL_SERVER_ERROR',
      message: 'An unexpected error occurred while processing your request.',
    },
  });
};

module.exports = errorHandler;
