/**
 * Real End-to-End Integration Tests.
 * Tests full flow: HTTP Client -> Express Backend -> Python FastAPI ML Service -> Scikit-learn Model -> MongoDB.
 * Section 75 & 86 compliant: Real services, no mocks.
 */

const request = require('supertest');
const mongoose = require('mongoose');
const app = require('../src/app');
const { connectDB, disconnectDB } = require('../src/config/database');
const Prediction = require('../src/models/Prediction');

describe('Full End-to-End System Integration Tests', () => {
  beforeAll(async () => {
    // Connect to MongoDB
    await connectDB();
  });

  afterAll(async () => {
    // Cleanup created test records
    await Prediction.deleteMany({ text: /TEST_E2E_/ });
    await disconnectDB();
  });

  test('E2E: GET /api/health confirms backend, ML service, and MongoDB connectivity', async () => {
    const response = await request(app).get('/api/health');

    expect(response.status).toBe(200);
    expect(response.body.success).toBe(true);
    expect(response.body.service).toBe('backend');
    expect(response.body.status).toBe('healthy');
    expect(response.body.components.mlService.status).toBe('available');
    expect(response.body.components.mlService.modelLoaded).toBe(true);
    expect(response.body.components.database.status).toBe('connected');
  });

  test('E2E: POST /api/predict executes live model inference and persists to MongoDB', async () => {
    const testText = 'TEST_E2E_ What a stunningly wonderful outcome for our hard work.';
    const response = await request(app)
      .post('/api/predict')
      .send({ text: testText });

    expect(response.status).toBe(200);
    expect(response.body.success).toBe(true);

    const data = response.body.data;
    expect(data.text).toBe(testText);
    expect(['Regular', 'Sarcasm', 'Irony', 'Figurative']).toContain(data.predicted_class);
    expect(data.confidence).toBeGreaterThanOrEqual(0.0);
    expect(data.confidence).toBeLessThanOrEqual(1.0);
    expect(data.probabilities).toHaveProperty('Regular');
    expect(data.probabilities).toHaveProperty('Sarcasm');
    expect(data.probabilities).toHaveProperty('Irony');
    expect(data.probabilities).toHaveProperty('Figurative');
    expect(data.model_name).toBe('SarcasmIronyClassifier-LinearSVM');
    expect(data.model_version).toBe('SIC-v1.0');
    expect(data.history_saved).toBe(true);
    expect(data.id).toBeDefined();

    // Verify record exists directly in MongoDB
    const dbDoc = await Prediction.findById(data.id);
    expect(dbDoc).not.toBeNull();
    expect(dbDoc.text).toBe(testText);
    expect(dbDoc.predictedClass).toBe(data.predicted_class);
    expect(dbDoc.modelVersion).toBe('SIC-v1.0');
  });

  test('E2E: GET /api/predictions retrieves stored predictions with pagination', async () => {
    const response = await request(app).get('/api/predictions?limit=5&page=1');

    expect(response.status).toBe(200);
    expect(response.body.success).toBe(true);
    expect(Array.isArray(response.body.data)).toBe(true);
    expect(response.body.data.length).toBeGreaterThan(0);
    expect(response.body.pagination).toBeDefined();
    expect(response.body.pagination.limit).toBe(5);
    expect(response.body.pagination.page).toBe(1);

    const firstItem = response.body.data[0];
    expect(firstItem.id).toBeDefined();
    expect(firstItem.text).toBeDefined();
    expect(firstItem.predicted_class).toBeDefined();
    expect(firstItem.modelVersion).toBe('SIC-v1.0');
  });

  test('E2E: POST /api/predict/batch executes live batch prediction', async () => {
    const batchTexts = [
      'TEST_E2E_ Beautiful sunny morning.',
      'TEST_E2E_ Thanks for breaking the code right before release.',
    ];

    const response = await request(app)
      .post('/api/predict/batch')
      .send({ texts: batchTexts });

    expect(response.status).toBe(200);
    expect(response.body.success).toBe(true);
    expect(response.body.data.total_count).toBe(2);
    expect(response.body.data.predictions.length).toBe(2);
    expect(response.body.data.batch_latency_ms).toBeGreaterThan(0);
  });

  test('E2E: DELETE /api/predictions/:id removes record from database', async () => {
    // Create a temporary record
    const record = await Prediction.create({
      text: 'TEST_E2E_ To be deleted',
      predictedClass: 'Regular',
      confidence: 0.9,
      probabilities: { Regular: 0.9, Sarcasm: 0.05, Irony: 0.03, Figurative: 0.02 },
      modelName: 'SarcasmIronyClassifier-LinearSVM',
      modelVersion: 'SIC-v1.0',
    });

    const response = await request(app).delete(`/api/predictions/${record.id}`);

    expect(response.status).toBe(200);
    expect(response.body.success).toBe(true);

    // Verify it is gone
    const checkDoc = await Prediction.findById(record.id);
    expect(checkDoc).toBeNull();
  });
});
