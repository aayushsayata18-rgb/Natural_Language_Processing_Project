import React from 'react';
import {
  formatPercentage,
  getClassDetails,
} from '../utils/formatting';
import ProbabilityChart from './ProbabilityChart';
import {
  Sparkles,
  Info,
  CheckCircle,
  HelpCircle,
  Zap,
  RotateCcw,
  AlertTriangle,
} from 'lucide-react';

export function PredictionResult({
  prediction,
  isStale = false,
  onReset,
  onReanalyze,
}) {
  if (!prediction) return null;

  const predictedClass = prediction.predicted_class || prediction.predictedClass || 'Regular';
  const confidence = prediction.confidence ?? 0;
  const probabilities = prediction.probabilities || {};
  const modelName = prediction.model_name || prediction.modelName || 'LinearSVM';
  const modelVersion = prediction.model_version || prediction.modelVersion || 'SIC-v1.0';
  const latencyMs = prediction.latency_ms ?? prediction.latencyMs;

  const classDetails = getClassDetails(predictedClass);

  return (
    <div
      className="card animate-fade-in"
      style={{
        border: `1px solid ${classDetails.hex}40`,
        position: 'relative',
        overflow: 'hidden',
      }}
      role="region"
      aria-label="Prediction Result"
    >
      {/* Subtle background glow */}
      <div
        style={{
          position: 'absolute',
          top: '-60px',
          right: '-60px',
          width: '180px',
          height: '180px',
          borderRadius: '50%',
          background: classDetails.hex,
          opacity: 0.1,
          filter: 'blur(50px)',
          pointerEvents: 'none',
        }}
      />

      {/* Stale Warning Banner (Section 50) */}
      {isStale && (
        <div
          style={{
            background: 'rgba(245, 158, 11, 0.12)',
            border: '1px solid rgba(245, 158, 11, 0.3)',
            borderRadius: '8px',
            padding: '0.65rem 1rem',
            marginBottom: '1.25rem',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '0.75rem',
            fontSize: '0.875rem',
            color: '#fbbf24',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <AlertTriangle size={16} />
            <span>The text above has been modified since this prediction was generated.</span>
          </div>
          {onReanalyze && (
            <button
              onClick={onReanalyze}
              className="btn btn-sm btn-secondary"
              style={{ padding: '0.25rem 0.6rem', fontSize: '0.75rem' }}
            >
              Update Result
            </button>
          )}
        </div>
      )}

      {/* Header section with primary prediction */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          flexWrap: 'wrap',
          gap: '1rem',
          paddingBottom: '1.25rem',
          borderBottom: '1px solid var(--border-color)',
        }}
      >
        <div>
          <span
            style={{
              fontSize: '0.75rem',
              textTransform: 'uppercase',
              letterSpacing: '0.06em',
              color: '#94a3b8',
              fontWeight: 600,
              display: 'flex',
              alignItems: 'center',
              gap: '0.35rem',
              marginBottom: '0.35rem',
            }}
          >
            <Sparkles size={14} style={{ color: classDetails.hex }} />
            NLP Classification Result
          </span>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
            <h2
              style={{
                fontSize: '1.85rem',
                color: '#ffffff',
                fontFamily: 'var(--font-heading)',
              }}
            >
              Detected: <span style={{ color: classDetails.hex }}>{predictedClass}</span>
            </h2>
            <span className={`badge ${classDetails.badgeClass}`}>
              {predictedClass}
            </span>
          </div>
        </div>

        {/* Confidence metric card */}
        <div
          style={{
            background: 'rgba(255, 255, 255, 0.04)',
            border: '1px solid var(--border-color)',
            borderRadius: '10px',
            padding: '0.65rem 1.25rem',
            textAlign: 'right',
          }}
        >
          <div style={{ fontSize: '0.75rem', color: '#94a3b8', textTransform: 'uppercase' }}>
            Model Confidence
          </div>
          <div
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '1.45rem',
              fontWeight: 700,
              color: classDetails.hex,
            }}
          >
            {formatPercentage(confidence, 1)}
          </div>
        </div>
      </div>

      {/* Result explanation text (Section 23) */}
      <div style={{ margin: '1.25rem 0 1.5rem' }}>
        <p style={{ color: '#cbd5e1', fontSize: '0.95rem', lineHeight: 1.6 }}>
          The model classified this text as{' '}
          <strong style={{ color: classDetails.hex }}>{predictedClass}</strong> with a model
          confidence of <strong>{formatPercentage(confidence, 1)}</strong> based on learned
          linguistic patterns.
        </p>
        <p style={{ color: '#94a3b8', fontSize: '0.85rem', marginTop: '0.35rem' }}>
          {classDetails.description}
        </p>
      </div>

      {/* Probabilities Distribution Chart */}
      <div
        style={{
          background: 'rgba(0, 0, 0, 0.25)',
          borderRadius: '12px',
          padding: '1.25rem',
          border: '1px solid rgba(255, 255, 255, 0.04)',
          marginBottom: '1.5rem',
        }}
      >
        <ProbabilityChart
          probabilities={probabilities}
          predictedClass={predictedClass}
        />
      </div>

      {/* Model metadata & performance tags */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '0.75rem',
          fontSize: '0.8rem',
          color: '#64748b',
          fontFamily: 'var(--font-mono)',
          paddingTop: '0.75rem',
          borderTop: '1px solid var(--border-color)',
          marginBottom: '1.25rem',
        }}
      >
        <div>
          Model: <span style={{ color: '#94a3b8' }}>{modelName} ({modelVersion})</span>
        </div>
        {latencyMs !== null && latencyMs !== undefined && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
            <Zap size={13} style={{ color: '#38bdf8' }} />
            Inference Latency: <span style={{ color: '#94a3b8' }}>{latencyMs} ms</span>
          </div>
        )}
      </div>

      {/* Academic Disclaimer (Section 24) */}
      <div
        style={{
          background: 'rgba(255, 255, 255, 0.02)',
          border: '1px solid rgba(255, 255, 255, 0.05)',
          borderRadius: '8px',
          padding: '0.75rem 1rem',
          display: 'flex',
          alignItems: 'flex-start',
          gap: '0.65rem',
          fontSize: '0.8rem',
          color: '#94a3b8',
          lineHeight: 1.5,
          marginBottom: '1.25rem',
        }}
      >
        <Info size={16} style={{ color: '#38bdf8', marginTop: '2px', flexShrink: 0 }} />
        <span>
          <strong>Ethical & Academic Note:</strong> Sarcasm and irony can be context-dependent and
          subjective. The prediction is an NLP model output and may not always reflect the author's
          intended meaning.
        </span>
      </div>

      {/* Actions */}
      <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'flex-end' }}>
        {onReset && (
          <button
            onClick={onReset}
            className="btn btn-secondary btn-sm"
            aria-label="Analyze another text"
          >
            <RotateCcw size={14} /> Analyze Another
          </button>
        )}
      </div>
    </div>
  );
}

export default PredictionResult;
