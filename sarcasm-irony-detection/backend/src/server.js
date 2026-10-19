/**
 * Application Entry Point.
 * Initializes database connection and starts Express HTTP server.
 */

const app = require('./app');
const config = require('./config/env');
const { connectDB, disconnectDB } = require('./config/database');

const startServer = async () => {
  // Connect to MongoDB (non-blocking for server startup if DB is down)
  await connectDB();

  const server = app.listen(config.port, () => {
    console.log('====================================================');
    console.log(`[Backend] Sarcasm & Irony Detection API Server running`);
    console.log(`[Backend] Environment : ${config.nodeEnv}`);
    console.log(`[Backend] Local URL   : http://localhost:${config.port}`);
    console.log(`[Backend] ML Service  : ${config.mlServiceUrl}`);
    console.log(`[Backend] Health Check: http://localhost:${config.port}/api/health`);
    console.log('====================================================');
  });

  // Graceful shutdown handling
  const shutdown = async (signal) => {
    console.log(`\n[Backend] Received ${signal}. Initiating graceful shutdown...`);
    server.close(async () => {
      console.log('[Backend] HTTP server closed.');
      await disconnectDB();
      console.log('[Backend] Clean shutdown complete.');
      process.exit(0);
    });

    // Force exit if shutdown takes too long
    setTimeout(() => {
      console.error('[Backend] Forced exit after timeout.');
      process.exit(1);
    }, 10000);
  };

  process.on('SIGTERM', () => shutdown('SIGTERM'));
  process.on('SIGINT', () => shutdown('SIGINT'));

  return server;
};

if (require.main === module) {
  startServer().catch((err) => {
    console.error('[Backend] Failed to start server:', err);
    process.exit(1);
  });
}

module.exports = { startServer };
