/**
 * Health Routes Tests using Supertest.
 */

const request = require('supertest');
const app = require('../src/app');
const mlService = require('../src/services/mlService');

jest.mock('../src/services/mlService');

describe('Health Routes', () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  test('GET /api/health should return healthy status when ML service is online', async () => {
    mlService.checkHealth.mockResolvedValue({
      status: 'ok',
      service: 'ml-service',
      model_loaded: true,
      model_name: 'SarcasmIronyClassifier-LinearSVM',
      model_version: 'SIC-v1.0',
    });

    const response = await request(app).get('/api/health');

    expect(response.status).toBe(200);
    expect(response.body.success).toBe(true);
    expect(response.body.service).toBe('backend');
    expect(response.body.status).toBe('healthy');
    expect(response.body.components.mlService.status).toBe('available');
    expect(response.body.components.mlService.modelLoaded).toBe(true);
  });

  test('GET /api/health should report degraded when ML service is offline', async () => {
    mlService.checkHealth.mockResolvedValue({
      status: 'error',
      service: 'ml-service',
      model_loaded: false,
    });

    const response = await request(app).get('/api/health');

    expect(response.status).toBe(200);
    expect(response.body.success).toBe(true);
    expect(response.body.status).toBe('degraded');
    expect(response.body.components.mlService.status).toBe('unavailable');
    expect(response.body.components.mlService.modelLoaded).toBe(false);
  });

  test('GET /api/model-info should return dynamic model metadata when ML service responds', async () => {
    mlService.getModelInfo.mockResolvedValue({
      model_name: 'SarcasmIronyClassifier-LinearSVM',
      model_version: 'SIC-v1.0',
      model_type: 'LinearSVC',
      classes: ['Regular', 'Sarcasm', 'Irony', 'Figurative'],
      macro_f1: 0.7719,
      accuracy: 0.77,
    });

    const response = await request(app).get('/api/model-info');

    expect(response.status).toBe(200);
    expect(response.body.success).toBe(true);
    expect(response.body.data.model_name).toBe('SarcasmIronyClassifier-LinearSVM');
    expect(response.body.data.macro_f1).toBe(0.7719);
  });

  test('GET /model-info root route should return fallback metadata when ML service fails', async () => {
    mlService.getModelInfo.mockRejectedValue(new Error('Down'));

    const response = await request(app).get('/model-info');

    expect(response.status).toBe(200);
    expect(response.body.success).toBe(true);
    expect(response.body.data.model_name).toBe('SarcasmIronyClassifier-LinearSVM');
    expect(response.body.data.status).toBe('cached_fallback');
  });
});

