import React from 'react';
import { AlertCircle, RotateCcw } from 'lucide-react';

export function ErrorMessage({ message, onRetry, title = 'Analysis Error' }) {
  if (!message) return null;

  return (
    <div
      role="alert"
      className="card animate-fade-in"
      style={{
        border: '1px solid rgba(239, 68, 68, 0.35)',
        background: 'rgba(239, 68, 68, 0.08)',
        display: 'flex',
        alignItems: 'flex-start',
        gap: '1rem',
        padding: '1.25rem 1.5rem',
      }}
    >
      <div
        style={{
          color: '#ef4444',
          marginTop: '2px',
          flexShrink: 0,
        }}
        aria-hidden="true"
      >
        <AlertCircle size={22} />
      </div>

      <div style={{ flex: 1 }}>
        <h4 style={{ color: '#f87171', fontSize: '1rem', marginBottom: '0.25rem' }}>
          {title}
        </h4>
        <p style={{ color: '#cbd5e1', fontSize: '0.9rem', lineHeight: 1.5 }}>
          {message}
        </p>
      </div>

      {onRetry && (
        <button
          onClick={onRetry}
          className="btn btn-secondary btn-sm"
          style={{ flexShrink: 0 }}
          aria-label="Retry action"
        >
          <RotateCcw size={14} /> Retry
        </button>
      )}
    </div>
  );
}

export default ErrorMessage;
