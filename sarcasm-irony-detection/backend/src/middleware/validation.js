/**
 * Request validation middleware.
 * Validates inputs prior to reaching application controllers or the ML service.
 */

const validatePredictRequest = (req, res, next) => {
  const { text } = req.body;

  if (text === undefined || text === null) {
    return res.status(400).json({
      success: false,
      error: {
        code: 'MISSING_REQUIRED_FIELD',
        message: 'Request body must contain a "text" field.',
      },
    });
  }

  if (typeof text !== 'string') {
    return res.status(400).json({
      success: false,
      error: {
        code: 'INVALID_DATA_TYPE',
        message: 'The "text" field must be a valid string.',
      },
    });
  }

  const trimmed = text.trim();
  if (trimmed.length === 0) {
    return res.status(400).json({
      success: false,
      error: {
        code: 'EMPTY_TEXT_INPUT',
        message: 'The "text" field cannot be empty or contain only whitespace.',
      },
    });
  }

  if (trimmed.length > 5000) {
    return res.status(400).json({
      success: false,
      error: {
        code: 'TEXT_TOO_LONG',
        message: `Input text exceeds the maximum permitted length of 5000 characters (received ${trimmed.length}).`,
      },
    });
  }

  if (req.body.model !== undefined && req.body.model !== null) {
    if (typeof req.body.model !== 'string') {
      return res.status(400).json({
        success: false,
        error: {
          code: 'INVALID_DATA_TYPE',
          message: 'The "model" field must be a valid string identifier.',
        },
      });
    }
    req.body.model = req.body.model.trim();
  }

  req.body.text = trimmed;
  next();
};

const validateBatchPredictRequest = (req, res, next) => {
  const { texts, model } = req.body;

  if (model !== undefined && model !== null) {
    if (typeof model !== 'string') {
      return res.status(400).json({
        success: false,
        error: {
          code: 'INVALID_DATA_TYPE',
          message: 'The "model" field must be a valid string identifier.',
        },
      });
    }
    req.body.model = model.trim();
  }

  if (!Array.isArray(texts)) {
    return res.status(400).json({
      success: false,
      error: {
        code: 'INVALID_DATA_TYPE',
        message: 'The "texts" field must be an array of strings.',
      },
    });
  }

  if (texts.length === 0) {
    return res.status(400).json({
      success: false,
      error: {
        code: 'EMPTY_BATCH',
        message: 'The "texts" array cannot be empty.',
      },
    });
  }

  if (texts.length > 100) {
    return res.status(400).json({
      success: false,
      error: {
        code: 'BATCH_TOO_LARGE',
        message: `Batch size exceeds maximum limit of 100 items (received ${texts.length}).`,
      },
    });
  }

  const cleaned = [];
  for (let i = 0; i < texts.length; i++) {
    const item = texts[i];
    if (typeof item !== 'string') {
      return res.status(400).json({
        success: false,
        error: {
          code: 'INVALID_BATCH_ITEM',
          message: `Item at index ${i} must be a string.`,
        },
      });
    }

    const trimmed = item.trim();
    if (trimmed.length === 0) {
      return res.status(400).json({
        success: false,
        error: {
          code: 'EMPTY_BATCH_ITEM',
          message: `Item at index ${i} cannot be empty or whitespace only.`,
        },
      });
    }

    if (trimmed.length > 5000) {
      return res.status(400).json({
        success: false,
        error: {
          code: 'BATCH_ITEM_TOO_LONG',
          message: `Item at index ${i} exceeds maximum permitted length of 5000 characters.`,
        },
      });
    }

    cleaned.push(trimmed);
  }

  req.body.texts = cleaned;
  next();
};

const validateHistoryQuery = (req, res, next) => {
  let limit = parseInt(req.query.limit, 10);
  let page = parseInt(req.query.page, 10);

  if (isNaN(limit) || limit < 1) {
    limit = 20;
  } else if (limit > 100) {
    limit = 100;
  }

  if (isNaN(page) || page < 1) {
    page = 1;
  }

  req.query.limit = limit;
  req.query.page = page;

  const validClasses = ['Regular', 'Sarcasm', 'Irony', 'Figurative'];
  if (req.query.predictedClass && !validClasses.includes(req.query.predictedClass)) {
    return res.status(400).json({
      success: false,
      error: {
        code: 'INVALID_QUERY_PARAM',
        message: `predictedClass query parameter must be one of: ${validClasses.join(', ')}`,
      },
    });
  }

  next();
};

module.exports = {
  validatePredictRequest,
  validateBatchPredictRequest,
  validateHistoryQuery,
};
