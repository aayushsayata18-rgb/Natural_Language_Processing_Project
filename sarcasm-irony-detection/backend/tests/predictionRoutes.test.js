/**
 * Integration tests for Prediction API Routes using Supertest.
 */

const request = require('supertest');
const app = require('../src/app');
const mlService = require('../src/services/mlService');
const { MLServiceError } = require('../src/services/mlService');

jest.mock('../src/services/mlService', () => {
  const actual = jest.requireActual('../src/services/mlService');
  return {
    ...actual,
    predict: jest.fn(),
    predictBatch: jest.fn(),
    checkHealth: jest.fn(),
    isReady: jest.fn(),
  };
});

describe('Prediction API Routes', () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('POST /api/predict', () => {
    test('should return 200 and formatted prediction on valid input', async () => {
      mlService.predict.mockResolvedValue({
        text: 'Just what I needed, more work.',
        predictedClass: 'Sarcasm',
        confidence: 0.912,
        probabilities: {
          Regular: 0.02,
          Sarcasm: 0.912,
          Irony: 0.05,
          Figurative: 0.018,
        },
        modelName: 'SarcasmIronyClassifier-LinearSVM',
        modelVersion: 'SIC-v1.0',
        latencyMs: 1.2,
      });

      const response = await request(app)
        .post('/api/predict')
        .send({ text: 'Just what I needed, more work.' });

      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.data).toMatchObject({
        text: 'Just what I needed, more work.',
        predicted_class: 'Sarcasm',
        confidence: 0.912,
        model_name: 'SarcasmIronyClassifier-LinearSVM',
        model_version: 'SIC-v1.0',
      });
      expect(response.body.data.probabilities).toHaveProperty('Sarcasm');
    });

    test('should return 400 when text is empty', async () => {
      const response = await request(app)
        .post('/api/predict')
        .send({ text: '   ' });

      expect(response.status).toBe(400);
      expect(response.body.success).toBe(false);
      expect(response.body.error.code).toBe('EMPTY_TEXT_INPUT');
    });

    test('should return 400 when text is missing', async () => {
      const response = await request(app).post('/api/predict').send({});

      expect(response.status).toBe(400);
      expect(response.body.success).toBe(false);
      expect(response.body.error.code).toBe('MISSING_REQUIRED_FIELD');
    });

    test('should return 503 when ML service is unavailable', async () => {
      mlService.predict.mockRejectedValue(
        new MLServiceError('ML service unreachable.', 'ML_SERVICE_UNAVAILABLE', 503)
      );

      const response = await request(app)
        .post('/api/predict')
        .send({ text: 'Valid sentence.' });

      expect(response.status).toBe(503);
      expect(response.body.success).toBe(false);
      expect(response.body.error.code).toBe('ML_SERVICE_UNAVAILABLE');
    });

    test('should return 504 when ML service times out', async () => {
      mlService.predict.mockRejectedValue(
        new MLServiceError('Request timed out.', 'ML_SERVICE_TIMEOUT', 504)
      );

      const response = await request(app)
        .post('/api/predict')
        .send({ text: 'Valid sentence.' });

      expect(response.status).toBe(504);
      expect(response.body.success).toBe(false);
      expect(response.body.error.code).toBe('ML_SERVICE_TIMEOUT');
    });
  });

  describe('GET /api/predictions', () => {
    test('should return 200 with prediction list structure', async () => {
      const response = await request(app).get('/api/predictions?limit=10&page=1');

      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(Array.isArray(response.body.data)).toBe(true);
      expect(response.body.pagination).toBeDefined();
    });
  });

  describe('DELETE /api/predictions/:id', () => {
    test('should return 400 for invalid ObjectId format', async () => {
      const response = await request(app).delete('/api/predictions/not-a-valid-id');

      expect(response.status).toBe(400);
      expect(response.body.success).toBe(false);
      expect(response.body.error.code).toBe('INVALID_IDENTIFIER');
    });
  });

  describe('404 Route Handling', () => {
    test('should return 404 for unmapped endpoint', async () => {
      const response = await request(app).get('/api/unmapped-endpoint-path');

      expect(response.status).toBe(404);
      expect(response.body.success).toBe(false);
      expect(response.body.error.code).toBe('NOT_FOUND');
    });
  });
});
