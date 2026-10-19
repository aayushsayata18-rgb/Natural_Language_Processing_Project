import React from 'react';
import { Loader2 } from 'lucide-react';

export function LoadingState({ message = 'Analyzing text with NLP pipeline...', size = 32 }) {
  return (
    <div
      role="status"
      aria-live="polite"
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '3rem 1.5rem',
        gap: '1rem',
      }}
      className="animate-fade-in"
    >
      <Loader2
        size={size}
        className="animate-spin"
        style={{ color: '#6366f1' }}
        aria-hidden="true"
      />
      <p style={{ color: '#94a3b8', fontSize: '0.95rem', fontWeight: 500 }}>
        {message}
      </p>
    </div>
  );
}

export default LoadingState;
