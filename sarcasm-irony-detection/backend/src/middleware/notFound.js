/**
 * Not Found (404) Route Middleware.
 * Returns standardized JSON for unmapped endpoints.
 */

const notFound = (req, res, next) => {
  res.status(404).json({
    success: false,
    error: {
      code: 'NOT_FOUND',
      message: `Cannot ${req.method} ${req.originalUrl}. Route not found.`,
    },
  });
};

module.exports = notFound;
