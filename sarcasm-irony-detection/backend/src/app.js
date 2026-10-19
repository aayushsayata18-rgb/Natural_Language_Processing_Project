/**
 * Express Application Configuration.
 * Configures security, CORS, request parsing, routing, and error handlers.
 */

const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');
const rateLimit = require('express-rate-limit');

const config = require('./config/env');
const healthRoutes = require('./routes/healthRoutes');
const predictionRoutes = require('./routes/predictionRoutes');
const notFound = require('./middleware/notFound');
const errorHandler = require('./middleware/errorHandler');

const app = express();

// Security HTTP headers
app.use(helmet());

// Cross-Origin Resource Sharing (CORS)
const corsOptions = {
  origin: (origin, callback) => {
    // Allow non-browser clients (like Postman or curl) or configured origin
    if (!origin || origin === config.corsOrigin || config.nodeEnv === 'development') {
      return callback(null, true);
    }
    return callback(new Error('CORS request blocked by security policy.'));
  },
  credentials: true,
  methods: ['GET', 'POST', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization'],
};
app.use(cors(corsOptions));

// Request body JSON parser
app.use(express.json({ limit: '1mb' }));
app.use(express.urlencoded({ extended: true, limit: '1mb' }));

// Logging: Logs method, url, status, and response-time without sensitive body text (Section 54)
if (config.nodeEnv !== 'test') {
  app.use(morgan(':method :url :status :res[content-length] - :response-time ms'));
}

// Basic rate limiting on prediction endpoints (Section 52)
const predictionLimiter = rateLimit({
  windowMs: config.rateLimitWindowMs,
  max: config.rateLimitMax,
  standardHeaders: true,
  legacyHeaders: false,
  message: {
    success: false,
    error: {
      code: 'TOO_MANY_REQUESTS',
      message: 'Too many requests from this IP. Please wait before retrying.',
    },
  },
});

// Mount Routes
app.use('/api', healthRoutes);
app.use('/', healthRoutes);
app.use('/api', predictionLimiter, predictionRoutes);

// Fallback for unmapped routes
app.use(notFound);

// Centralized error handler
app.use(errorHandler);

module.exports = app;
