/**
 * Unit tests for ML Service Client.
 */

const axios = require('axios');
const { MLService, MLServiceError } = require('../src/services/mlService');

jest.mock('axios');

describe('MLService Client Tests', () => {
  let mlService;
  let mockAxiosInstance;

  beforeEach(() => {
    mockAxiosInstance = {
      get: jest.fn(),
      post: jest.fn(),
    };
    axios.create.mockReturnValue(mockAxiosInstance);
    mlService = new MLService();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('predict', () => {
    test('should successfully parse and format a valid ML prediction', async () => {
      const mockFastApiResponse = {
        data: {
          text: 'Oh great, another delay.',
          predicted_class: 'Sarcasm',
          confidence: 0.8921,
          probabilities: {
            Regular: 0.05,
            Sarcasm: 0.8921,
            Irony: 0.03,
            Figurative: 0.0279,
          },
          model_name: 'SarcasmIronyClassifier-LinearSVM',
          model_version: 'SIC-v1.0',
          latency_ms: 1.45,
        },
      };

      mockAxiosInstance.post.mockResolvedValue(mockFastApiResponse);

      const result = await mlService.predict('Oh great, another delay.');

      expect(mockAxiosInstance.post).toHaveBeenCalledWith('/predict', {
        text: 'Oh great, another delay.',
      });
      expect(result).toEqual({
        text: 'Oh great, another delay.',
        predictedClass: 'Sarcasm',
        confidence: 0.8921,
        probabilities: {
          Regular: 0.05,
          Sarcasm: 0.8921,
          Irony: 0.03,
          Figurative: 0.0279,
        },
        modelName: 'SarcasmIronyClassifier-LinearSVM',
        modelVersion: 'SIC-v1.0',
        latencyMs: 1.45,
      });
    });

    test('should reject invalid upstream contract (missing predicted_class)', async () => {
      const invalidResponse = {
        data: {
          text: 'Some text',
          confidence: 0.5,
          probabilities: { Regular: 0.5, Sarcasm: 0.5, Irony: 0, Figurative: 0 },
        },
      };

      mockAxiosInstance.post.mockResolvedValue(invalidResponse);

      await expect(mlService.predict('Some text')).rejects.toThrow(MLServiceError);
      await expect(mlService.predict('Some text')).rejects.toMatchObject({
        code: 'INVALID_UPSTREAM_RESPONSE',
        statusCode: 502,
      });
    });

    test('should handle connection timeout and throw ML_SERVICE_TIMEOUT', async () => {
      const timeoutError = new Error('timeout of 5000ms exceeded');
      timeoutError.code = 'ECONNABORTED';

      mockAxiosInstance.post.mockRejectedValue(timeoutError);

      await expect(mlService.predict('Some text')).rejects.toMatchObject({
        code: 'ML_SERVICE_TIMEOUT',
        statusCode: 504,
      });
    });

    test('should handle service down and throw ML_SERVICE_UNAVAILABLE', async () => {
      const connRefusedError = new Error('connect ECONNREFUSED 127.0.0.1:8000');
      connRefusedError.code = 'ECONNREFUSED';

      mockAxiosInstance.post.mockRejectedValue(connRefusedError);

      await expect(mlService.predict('Some text')).rejects.toMatchObject({
        code: 'ML_SERVICE_UNAVAILABLE',
        statusCode: 503,
      });
    });

    test('should propagate 422 validation error from Python service', async () => {
      const validationError = {
        response: {
          status: 422,
          data: { error: 'ValidationError', message: 'Input text too long' },
        },
      };

      mockAxiosInstance.post.mockRejectedValue(validationError);

      await expect(mlService.predict('Some text')).rejects.toMatchObject({
        code: 'VALIDATION_ERROR',
        statusCode: 422,
        message: 'Input text too long',
      });
    });
  });

  describe('checkHealth', () => {
    test('should return health status when service is up', async () => {
      mockAxiosInstance.get.mockResolvedValue({
        data: { status: 'ok', service: 'ml-service', model_loaded: true },
      });

      const res = await mlService.checkHealth();
      expect(res.status).toBe('ok');
      expect(res.model_loaded).toBe(true);
    });

    test('should return error object gracefully when service is down', async () => {
      mockAxiosInstance.get.mockRejectedValue(new Error('Connection failed'));

      const res = await mlService.checkHealth();
      expect(res.status).toBe('error');
      expect(res.model_loaded).toBe(false);
    });
  });
});
