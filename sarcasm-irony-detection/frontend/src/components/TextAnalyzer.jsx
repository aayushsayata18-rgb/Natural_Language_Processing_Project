import React from 'react';
import { Send, Eraser, Sparkles, CornerDownLeft } from 'lucide-react';

export const EXAMPLE_TEXTS = [
  {
    label: 'Meeting',
    category: 'Sarcasm',
    text: 'Fantastic, another three-hour meeting on a Friday afternoon.',
  },
  {
    label: 'Weather',
    category: 'Regular',
    text: 'What a wonderful sunny day for a walk in the park.',
  },
  {
    label: 'Fire Station',
    category: 'Irony',
    text: 'The local fire station unexpectedly caught fire during the safety demonstration.',
  },
  {
    label: 'Metaphor',
    category: 'Figurative',
    text: 'Time is a thief that silently steals away our precious youthful years.',
  },
  {
    label: 'Software Bug',
    category: 'Sarcasm',
    text: 'Brilliant code! It only failed in every single automated test.',
  },
];

export function TextAnalyzer({
  inputText,
  onTextChange,
  onAnalyze,
  onClear,
  loading = false,
  maxLength = 5000,
}) {
  const charCount = inputText.length;
  const isOverLimit = charCount > maxLength;
  const isEmpty = inputText.trim().length === 0;

  const handleKeyDown = (e) => {
    // Ctrl/Cmd + Enter submits the text (Section 51)
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      if (!isEmpty && !loading && !isOverLimit) {
        onAnalyze();
      }
    }
  };

  const handleSelectExample = (exampleText) => {
    onTextChange(exampleText);
  };

  return (
    <div className="card" role="region" aria-label="Text Analysis Form">
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (!isEmpty && !loading && !isOverLimit) {
            onAnalyze();
          }
        }}
      >
        <div style={{ marginBottom: '0.75rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <label
            htmlFor="analyzer-textarea"
            style={{
              fontSize: '0.95rem',
              fontWeight: 600,
              color: '#e2e8f0',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
            }}
          >
            <Sparkles size={16} style={{ color: '#6366f1' }} />
            Enter Sentence or Social Media Text
          </label>

          <span
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '0.8rem',
              color: isOverLimit ? '#ef4444' : charCount > maxLength * 0.8 ? '#f59e0b' : '#64748b',
              fontWeight: isOverLimit ? 700 : 500,
            }}
            aria-live="polite"
          >
            {charCount} / {maxLength}
          </span>
        </div>

        {/* Textarea */}
        <div style={{ position: 'relative' }}>
          <textarea
            id="analyzer-textarea"
            value={inputText}
            onChange={(e) => onTextChange(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type or paste text here (e.g., 'Oh great, my flight is delayed by another 4 hours...')"
            rows={5}
            style={{
              width: '100%',
              backgroundColor: 'var(--bg-input)',
              border: `1px solid ${isOverLimit ? '#ef4444' : 'var(--border-color)'}`,
              borderRadius: 'var(--radius-md)',
              padding: '1rem',
              color: 'var(--text-primary)',
              fontFamily: 'var(--font-body)',
              fontSize: '1rem',
              lineHeight: 1.6,
              resize: 'vertical',
              minHeight: '120px',
              transition: 'border-color var(--transition-fast)',
            }}
            disabled={loading}
          />
        </div>

        {/* Example Quick-Pick Pills */}
        <div style={{ marginTop: '1rem', marginBottom: '1.25rem' }}>
          <div
            style={{
              fontSize: '0.8rem',
              color: '#94a3b8',
              marginBottom: '0.5rem',
              fontWeight: 500,
            }}
          >
            Try an example:
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
            {EXAMPLE_TEXTS.map((item, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => handleSelectExample(item.text)}
                disabled={loading}
                className="btn btn-secondary btn-sm"
                style={{
                  fontSize: '0.8rem',
                  padding: '0.3rem 0.7rem',
                  borderRadius: 'var(--radius-full)',
                  borderColor: 'rgba(255, 255, 255, 0.08)',
                }}
                title={item.text}
              >
                <span>{item.label}</span>
                <span
                  style={{
                    fontSize: '0.7rem',
                    color: '#94a3b8',
                    marginLeft: '0.25rem',
                    fontFamily: 'var(--font-mono)',
                  }}
                >
                  ({item.category})
                </span>
              </button>
            ))}
          </div>
        </div>

        {/* Controls: Clear & Submit */}
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: '0.75rem',
            paddingTop: '0.75rem',
            borderTop: '1px solid var(--border-color)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <button
              type="button"
              onClick={onClear}
              disabled={loading || isEmpty}
              className="btn btn-secondary btn-sm"
              aria-label="Clear text input"
            >
              <Eraser size={14} /> Clear
            </button>
            <span
              style={{
                fontSize: '0.75rem',
                color: '#64748b',
                display: 'none',
                alignItems: 'center',
                gap: '0.25rem',
              }}
              className="keyboard-hint"
            >
              <CornerDownLeft size={12} /> Press Ctrl+Enter to analyze
            </span>
          </div>

          <button
            type="submit"
            disabled={loading || isEmpty || isOverLimit}
            className="btn btn-primary"
            style={{ minWidth: '150px' }}
          >
            {loading ? (
              <>
                <span
                  className="animate-spin"
                  style={{
                    width: '14px',
                    height: '14px',
                    border: '2px solid rgba(255, 255, 255, 0.3)',
                    borderTopColor: '#ffffff',
                    borderRadius: '50%',
                    display: 'inline-block',
                  }}
                />
                Analyzing...
              </>
            ) : (
              <>
                <Send size={16} /> Analyze Text
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}

export default TextAnalyzer;
