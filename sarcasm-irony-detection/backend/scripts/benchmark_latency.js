/**
 * Latency Benchmark Script (Phase 11 - Section 77).
 * Measures:
 * 1. Node.js -> Python ML Service round-trip latency
 * 2. Python ML Service inference latency
 */

const axios = require('axios');

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';
const SAMPLE_TEXTS = [
  'Fantastic, another three-hour meeting on a Friday afternoon.',
  'The laboratory experiment yielded promising scientific results.',
  'Oh wonderful, the train broke down just when I had an interview.',
  'The research paper describes a multi-class text categorization system.',
  'Brilliant design! It only crashes whenever someone clicks anything.',
];

const NUM_REQUESTS = 50;

async function runBenchmark() {
  console.log('====================================================');
  console.log(`Starting Phase 11 Latency Benchmark (${NUM_REQUESTS} requests)`);
  console.log(`Target ML Service: ${ML_SERVICE_URL}`);
  console.log('====================================================');

  const roundTripLatencies = [];
  const modelLatencies = [];

  // Warm-up request
  try {
    await axios.post(`${ML_SERVICE_URL}/predict`, { text: 'Warmup test sentence.' });
  } catch (err) {
    console.error('Failed warmup request:', err.message);
    process.exit(1);
  }

  for (let i = 0; i < NUM_REQUESTS; i++) {
    const text = SAMPLE_TEXTS[i % SAMPLE_TEXTS.length];
    const start = performance.now();
    const res = await axios.post(`${ML_SERVICE_URL}/predict`, { text });
    const roundTrip = performance.now() - start;

    roundTripLatencies.push(roundTrip);
    if (res.data.latency_ms !== undefined) {
      modelLatencies.push(res.data.latency_ms);
    }
  }

  const calcStats = (arr) => {
    const sum = arr.reduce((a, b) => a + b, 0);
    const avg = sum / arr.length;
    const min = Math.min(...arr);
    const max = Math.max(...arr);
    const sorted = [...arr].sort((a, b) => a - b);
    const p95 = sorted[Math.floor(sorted.length * 0.95)];
    return { avg: avg.toFixed(3), min: min.toFixed(3), max: max.toFixed(3), p95: p95.toFixed(3) };
  };

  const rtStats = calcStats(roundTripLatencies);
  const mlStats = calcStats(modelLatencies);

  console.log('\n--- Benchmark Results ---');
  console.log(`Total Requests Completed : ${NUM_REQUESTS}`);
  console.log('\n1. Node -> Python Round-Trip HTTP Latency (ms):');
  console.log(`   - Average : ${rtStats.avg} ms`);
  console.log(`   - Min     : ${rtStats.min} ms`);
  console.log(`   - Max     : ${rtStats.max} ms`);
  console.log(`   - P95     : ${rtStats.p95} ms`);

  console.log('\n2. Python Internal Model Inference Latency (ms):');
  console.log(`   - Average : ${mlStats.avg} ms`);
  console.log(`   - Min     : ${mlStats.min} ms`);
  console.log(`   - Max     : ${mlStats.max} ms`);
  console.log(`   - P95     : ${mlStats.p95} ms`);
  console.log('====================================================\n');
}

runBenchmark().catch((err) => {
  console.error('Benchmark failed:', err);
  process.exit(1);
});
