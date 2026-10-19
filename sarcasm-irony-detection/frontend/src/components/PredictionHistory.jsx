import React, { useState } from 'react';
import {
  formatPercentage,
  formatDateTime,
  truncateText,
  getClassDetails,
} from '../utils/formatting';
import {
  Trash2,
  ChevronLeft,
  ChevronRight,
  Filter,
  Inbox,
  AlertCircle,
  ExternalLink,
} from 'lucide-react';
import { Link } from 'react-router-dom';

export function PredictionHistory({
  predictions = [],
  pagination = { total: 0, page: 1, limit: 20, pages: 1 },
  loading = false,
  error = null,
  onPageChange,
  onFilterChange,
  selectedClass = '',
  onDelete,
  onRetry,
}) {
  const [deletingId, setDeletingId] = useState(null);
  const [confirmDeleteId, setConfirmDeleteId] = useState(null);

  const classes = ['All', 'Regular', 'Sarcasm', 'Irony', 'Figurative'];

  const handleDeleteClick = async (id) => {
    setDeletingId(id);
    try {
      await onDelete(id);
    } finally {
      setDeletingId(null);
      setConfirmDeleteId(null);
    }
  };

  if (loading) {
    return (
      <div className="card" style={{ textAlign: 'center', padding: '3.5rem 1.5rem' }}>
        <span
          className="animate-spin"
          style={{
            width: '28px',
            height: '28px',
            border: '3px solid rgba(99, 102, 241, 0.2)',
            borderTopColor: '#6366f1',
            borderRadius: '50%',
            display: 'inline-block',
            marginBottom: '1rem',
          }}
        />
        <p style={{ color: '#94a3b8' }}>Loading prediction history from database...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card" style={{ padding: '2.5rem', textAlign: 'center' }}>
        <div
          style={{
            width: '48px',
            height: '48px',
            borderRadius: '50%',
            background: 'rgba(239, 68, 68, 0.15)',
            color: '#ef4444',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            margin: '0 auto 1rem',
          }}
        >
          <AlertCircle size={24} />
        </div>
        <h3 style={{ color: '#f8fafc', marginBottom: '0.5rem' }}>
          Unable to Load Prediction History
        </h3>
        <p style={{ color: '#94a3b8', fontSize: '0.95rem', marginBottom: '1.5rem', maxWidth: '450px', margin: '0 auto 1.5rem' }}>
          {error.message || 'Database connection could not be established or the service is temporarily offline.'}
        </p>
        {onRetry && (
          <button onClick={onRetry} className="btn btn-secondary">
            Retry Loading
          </button>
        )}
      </div>
    );
  }

  return (
    <div className="history-container" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      {/* Filter and summary bar */}
      <div
        className="card"
        style={{
          padding: '1rem 1.25rem',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '1rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
          <Filter size={18} style={{ color: '#6366f1' }} />
          <span style={{ fontSize: '0.9rem', color: '#cbd5e1', fontWeight: 500 }}>
            Filter by Class:
          </span>
          <div style={{ display: 'flex', gap: '0.35rem', flexWrap: 'wrap' }}>
            {classes.map((cls) => {
              const active = (cls === 'All' && !selectedClass) || selectedClass === cls;
              return (
                <button
                  key={cls}
                  onClick={() => onFilterChange(cls === 'All' ? '' : cls)}
                  className={`btn btn-sm ${active ? 'btn-primary' : 'btn-secondary'}`}
                  style={{ padding: '0.25rem 0.65rem', fontSize: '0.8rem' }}
                >
                  {cls}
                </button>
              );
            })}
          </div>
        </div>

        <div style={{ fontSize: '0.85rem', color: '#94a3b8', fontFamily: 'var(--font-mono)' }}>
          Total Stored: <strong>{pagination.total || 0}</strong>
        </div>
      </div>

      {/* Empty State */}
      {predictions.length === 0 ? (
        <div
          className="card"
          style={{
            textAlign: 'center',
            padding: '4rem 1.5rem',
          }}
        >
          <div
            style={{
              width: '56px',
              height: '56px',
              borderRadius: '50%',
              background: 'rgba(255, 255, 255, 0.04)',
              color: '#64748b',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              margin: '0 auto 1.25rem',
            }}
          >
            <Inbox size={28} />
          </div>
          <h3 style={{ fontSize: '1.25rem', color: '#f8fafc', marginBottom: '0.5rem' }}>
            No predictions yet
          </h3>
          <p style={{ color: '#94a3b8', fontSize: '0.95rem', marginBottom: '1.5rem' }}>
            Analyze a sentence to generate and view historical classification records.
          </p>
          <Link to="/detector" className="btn btn-primary">
            Analyze your first text
          </Link>
        </div>
      ) : (
        <>
          {/* Desktop Table View */}
          <div
            className="card"
            style={{
              padding: 0,
              overflow: 'hidden',
            }}
          >
            <div style={{ overflowX: 'auto' }}>
              <table
                style={{
                  width: '100%',
                  borderCollapse: 'collapse',
                  textAlign: 'left',
                  fontSize: '0.9rem',
                }}
              >
                <thead>
                  <tr
                    style={{
                      background: 'rgba(255, 255, 255, 0.02)',
                      borderBottom: '1px solid var(--border-color)',
                      color: '#94a3b8',
                      fontSize: '0.75rem',
                      textTransform: 'uppercase',
                      letterSpacing: '0.05em',
                    }}
                  >
                    <th style={{ padding: '0.85rem 1.25rem' }}>Input Text</th>
                    <th style={{ padding: '0.85rem 1rem' }}>Classification</th>
                    <th style={{ padding: '0.85rem 1rem' }}>Confidence</th>
                    <th style={{ padding: '0.85rem 1rem' }}>Model</th>
                    <th style={{ padding: '0.85rem 1rem' }}>Date & Time</th>
                    {onDelete && <th style={{ padding: '0.85rem 1rem', textAlign: 'right' }}>Actions</th>}
                  </tr>
                </thead>
                <tbody>
                  {predictions.map((p) => {
                    const cls = p.predictedClass || p.predicted_class || 'Regular';
                    const details = getClassDetails(cls);
                    const recordId = p.id || p._id;
                    const isConfirming = confirmDeleteId === recordId;
                    const isDeleting = deletingId === recordId;

                    return (
                      <tr
                        key={recordId}
                        style={{
                          borderBottom: '1px solid rgba(255, 255, 255, 0.04)',
                          transition: 'background var(--transition-fast)',
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.background = 'rgba(255, 255, 255, 0.02)';
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.background = 'transparent';
                        }}
                      >
                        <td
                          style={{
                            padding: '1rem 1.25rem',
                            maxWidth: '380px',
                            color: '#f8fafc',
                          }}
                        >
                          <span title={p.text}>{truncateText(p.text, 90)}</span>
                        </td>
                        <td style={{ padding: '1rem' }}>
                          <span className={`badge ${details.badgeClass}`}>{cls}</span>
                        </td>
                        <td
                          style={{
                            padding: '1rem',
                            fontFamily: 'var(--font-mono)',
                            color: details.hex,
                            fontWeight: 600,
                          }}
                        >
                          {formatPercentage(p.confidence, 1)}
                        </td>
                        <td
                          style={{
                            padding: '1rem',
                            fontSize: '0.8rem',
                            fontFamily: 'var(--font-mono)',
                            color: '#94a3b8',
                          }}
                        >
                          {p.modelVersion || p.model_version || 'SIC-v1.0'}
                        </td>
                        <td
                          style={{
                            padding: '1rem',
                            fontSize: '0.8rem',
                            color: '#64748b',
                            whiteSpace: 'nowrap',
                          }}
                        >
                          {formatDateTime(p.createdAt || p.created_at)}
                        </td>
                        {onDelete && (
                          <td style={{ padding: '1rem', textAlign: 'right' }}>
                            {isConfirming ? (
                              <div style={{ display: 'inline-flex', gap: '0.35rem' }}>
                                <button
                                  onClick={() => handleDeleteClick(recordId)}
                                  disabled={isDeleting}
                                  className="btn btn-danger btn-sm"
                                  style={{ padding: '0.2rem 0.5rem', fontSize: '0.75rem' }}
                                >
                                  {isDeleting ? 'Deleting...' : 'Confirm'}
                                </button>
                                <button
                                  onClick={() => setConfirmDeleteId(null)}
                                  className="btn btn-secondary btn-sm"
                                  style={{ padding: '0.2rem 0.5rem', fontSize: '0.75rem' }}
                                >
                                  Cancel
                                </button>
                              </div>
                            ) : (
                              <button
                                onClick={() => setConfirmDeleteId(recordId)}
                                className="btn btn-secondary btn-sm"
                                style={{
                                  padding: '0.3rem 0.5rem',
                                  color: '#ef4444',
                                  borderColor: 'transparent',
                                }}
                                aria-label="Delete prediction from history"
                                title="Delete prediction"
                              >
                                <Trash2 size={15} />
                              </button>
                            )}
                          </td>
                        )}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Pagination Controls (Section 32) */}
          {pagination.pages > 1 && (
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '0.5rem 0',
              }}
            >
              <button
                onClick={() => onPageChange(pagination.page - 1)}
                disabled={pagination.page <= 1}
                className="btn btn-secondary btn-sm"
              >
                <ChevronLeft size={16} /> Previous
              </button>

              <span
                style={{
                  fontSize: '0.85rem',
                  color: '#94a3b8',
                  fontFamily: 'var(--font-mono)',
                }}
              >
                Page {pagination.page} of {pagination.pages}
              </span>

              <button
                onClick={() => onPageChange(pagination.page + 1)}
                disabled={pagination.page >= pagination.pages}
                className="btn btn-secondary btn-sm"
              >
                Next <ChevronRight size={16} />
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default PredictionHistory;
