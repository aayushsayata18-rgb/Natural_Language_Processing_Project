import React from 'react';
import { formatPercentage, CLASS_COLORS } from '../utils/formatting';
import { BarChart3, AlertCircle } from 'lucide-react';

export function ProbabilityChart({ probabilities = {}, predictedClass = '' }) {
  const classes = ['Regular', 'Sarcasm', 'Irony', 'Figurative'];

  // Validation: Check if probabilities object is valid
  const isValid =
    probabilities &&
    typeof probabilities === 'object' &&
    classes.every((c) => typeof probabilities[c] === 'number' && !isNaN(probabilities[c]));

  if (!isValid) {
    return (
      <div
        style={{
          padding: '1.25rem',
          background: 'rgba(255, 255, 255, 0.03)',
          borderRadius: '8px',
          color: '#94a3b8',
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem',
          fontSize: '0.9rem',
        }}
      >
        <AlertCircle size={18} style={{ color: '#f59e0b' }} />
        <span>Prediction details are temporarily unavailable.</span>
      </div>
    );
  }

  return (
    <div
      className="probability-chart"
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '1rem',
      }}
      role="region"
      aria-label="Class probability distribution"
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: '0.25rem',
        }}
      >
        <h4
          style={{
            fontSize: '0.95rem',
            fontWeight: 600,
            color: '#cbd5e1',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
          }}
        >
          <BarChart3 size={16} style={{ color: '#38bdf8' }} /> Class Probabilities
        </h4>
        <span style={{ fontSize: '0.75rem', color: '#64748b' }}>
          Calibrated Softmax Distribution
        </span>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
        {classes.map((className) => {
          const rawProb = probabilities[className] || 0;
          const percentageStr = formatPercentage(rawProb, 1);
          // Scale to 0-100 for width
          const widthPercent = Math.min(100, Math.max(0, (rawProb <= 1 ? rawProb * 100 : rawProb)));
          const isWinner = className === predictedClass;
          const config = CLASS_COLORS[className];

          return (
            <div
              key={className}
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '0.35rem',
              }}
            >
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  fontSize: '0.875rem',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <span
                    style={{
                      width: '8px',
                      height: '8px',
                      borderRadius: '50%',
                      background: config.hex,
                      display: 'inline-block',
                    }}
                  />
                  <span
                    style={{
                      fontWeight: isWinner ? 600 : 400,
                      color: isWinner ? '#ffffff' : '#cbd5e1',
                    }}
                  >
                    {config.label}
                  </span>
                  {isWinner && (
                    <span
                      style={{
                        fontSize: '0.7rem',
                        padding: '1px 6px',
                        borderRadius: '4px',
                        background: config.bg,
                        color: config.hex,
                        border: `1px solid ${config.hex}40`,
                        fontWeight: 600,
                      }}
                    >
                      PREDICTED
                    </span>
                  )}
                </div>
                <span
                  style={{
                    fontFamily: 'var(--font-mono)',
                    fontWeight: isWinner ? 700 : 500,
                    color: isWinner ? config.hex : '#94a3b8',
                  }}
                >
                  {percentageStr}
                </span>
              </div>

              {/* Progress Track */}
              <div
                style={{
                  width: '100%',
                  height: '8px',
                  background: 'rgba(255, 255, 255, 0.05)',
                  borderRadius: '9999px',
                  overflow: 'hidden',
                  position: 'relative',
                }}
                role="progressbar"
                aria-valuenow={Math.round(widthPercent)}
                aria-valuemin="0"
                aria-valuemax="100"
                aria-label={`${className} probability`}
              >
                <div
                  style={{
                    height: '100%',
                    width: `${widthPercent}%`,
                    backgroundColor: config.hex,
                    borderRadius: '9999px',
                    transition: 'width 0.6s cubic-bezier(0.16, 1, 0.3, 1)',
                    boxShadow: isWinner ? `0 0 10px ${config.hex}80` : 'none',
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default ProbabilityChart;
