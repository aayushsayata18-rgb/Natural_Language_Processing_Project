/**
 * Unit tests for Request Validation Middleware.
 */

const {
  validatePredictRequest,
  validateBatchPredictRequest,
  validateHistoryQuery,
} = require('../src/middleware/validation');

describe('Validation Middleware Tests', () => {
  let req, res, next;

  beforeEach(() => {
    req = { body: {}, query: {} };
    res = {
      status: jest.fn().mockReturnThis(),
      json: jest.fn().mockReturnThis(),
    };
    next = jest.fn();
  });

  describe('validatePredictRequest', () => {
    test('should pass for valid text and trim whitespace', () => {
      req.body.text = '   This is a clean valid sentence.   ';
      validatePredictRequest(req, res, next);

      expect(next).toHaveBeenCalled();
      expect(req.body.text).toBe('This is a clean valid sentence.');
      expect(res.status).not.toHaveBeenCalled();
    });

    test('should reject missing text field with 400', () => {
      validatePredictRequest(req, res, next);

      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith(
        expect.objectContaining({
          success: false,
          error: expect.objectContaining({ code: 'MISSING_REQUIRED_FIELD' }),
        })
      );
      expect(next).not.toHaveBeenCalled();
    });

    test('should reject non-string text field with 400', () => {
      req.body.text = 12345;
      validatePredictRequest(req, res, next);

      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith(
        expect.objectContaining({
          success: false,
          error: expect.objectContaining({ code: 'INVALID_DATA_TYPE' }),
        })
      );
      expect(next).not.toHaveBeenCalled();
    });

    test('should reject empty or whitespace text with 400', () => {
      req.body.text = '    \n\t   ';
      validatePredictRequest(req, res, next);

      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith(
        expect.objectContaining({
          success: false,
          error: expect.objectContaining({ code: 'EMPTY_TEXT_INPUT' }),
        })
      );
      expect(next).not.toHaveBeenCalled();
    });

    test('should reject excessively long text (>5000 chars) with 400', () => {
      req.body.text = 'a'.repeat(5001);
      validatePredictRequest(req, res, next);

      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith(
        expect.objectContaining({
          success: false,
          error: expect.objectContaining({ code: 'TEXT_TOO_LONG' }),
        })
      );
      expect(next).not.toHaveBeenCalled();
    });
  });

  describe('validateBatchPredictRequest', () => {
    test('should pass for valid batch and trim texts', () => {
      req.body.texts = ['  Text 1  ', 'Text 2'];
      validateBatchPredictRequest(req, res, next);

      expect(next).toHaveBeenCalled();
      expect(req.body.texts).toEqual(['Text 1', 'Text 2']);
    });

    test('should reject non-array batch with 400', () => {
      req.body.texts = 'not-an-array';
      validateBatchPredictRequest(req, res, next);

      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith(
        expect.objectContaining({
          success: false,
          error: expect.objectContaining({ code: 'INVALID_DATA_TYPE' }),
        })
      );
    });

    test('should reject empty batch list with 400', () => {
      req.body.texts = [];
      validateBatchPredictRequest(req, res, next);

      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith(
        expect.objectContaining({
          success: false,
          error: expect.objectContaining({ code: 'EMPTY_BATCH' }),
        })
      );
    });

    test('should reject batch with empty element with 400', () => {
      req.body.texts = ['Valid text', '   '];
      validateBatchPredictRequest(req, res, next);

      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith(
        expect.objectContaining({
          success: false,
          error: expect.objectContaining({ code: 'EMPTY_BATCH_ITEM' }),
        })
      );
    });
  });

  describe('validateHistoryQuery', () => {
    test('should apply default pagination when omitted', () => {
      validateHistoryQuery(req, res, next);

      expect(next).toHaveBeenCalled();
      expect(req.query.limit).toBe(20);
      expect(req.query.page).toBe(1);
    });

    test('should clamp limit between 1 and 100', () => {
      req.query.limit = '500';
      validateHistoryQuery(req, res, next);

      expect(next).toHaveBeenCalled();
      expect(req.query.limit).toBe(100);
    });

    test('should reject invalid predictedClass query parameter', () => {
      req.query.predictedClass = 'InvalidClass';
      validateHistoryQuery(req, res, next);

      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith(
        expect.objectContaining({
          success: false,
          error: expect.objectContaining({ code: 'INVALID_QUERY_PARAM' }),
        })
      );
    });
  });
});
