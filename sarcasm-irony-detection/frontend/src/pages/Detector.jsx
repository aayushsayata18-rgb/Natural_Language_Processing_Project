import React, { useRef, useEffect } from 'react';
import usePrediction from '../hooks/usePrediction';
import TextAnalyzer from '../components/TextAnalyzer';
import PredictionResult from '../components/PredictionResult';
import LoadingState from '../components/LoadingState';
import ErrorMessage from '../components/ErrorMessage';
import { Sparkles, Cpu, AlertCircle } from 'lucide-react';

export function Detector() {
  const {
    inputText,
    setInputText,
    prediction,
    loading,
    error,
    isStale,
    analyze,
    reset,
    maxInputLength,
  } = usePrediction();

  const resultRef = useRef(null);

  // Auto-scroll result into view upon receiving successful prediction (Section 72)
  useEffect(() => {
    if (
      prediction &&
      resultRef.current &&
      typeof resultRef.current.scrollIntoView === 'function'
    ) {
      resultRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, [prediction]);


  return (
    <div
      className="detector-page"
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '2rem',
        maxWidth: '850px',
        margin: '0 auto',
      }}
    >
      {/* Header */}
      <div style={{ textAlign: 'center', marginBottom: '0.5rem' }}>
        <h1
          style={{
            fontSize: '2.25rem',
            color: '#ffffff',
            marginBottom: '0.5rem',
            fontFamily: 'var(--font-heading)',
          }}
        >
          Sarcasm & Irony Detector
        </h1>
        <p style={{ color: '#94a3b8', fontSize: '1rem', maxWidth: '600px', margin: '0 auto' }}>
          Evaluate textual discourse across Regular, Sarcasm, Irony, and Figurative language
          categories with calibrated confidence metrics.
        </p>
      </div>

      {/* Input Component */}
      <TextAnalyzer
        inputText={inputText}
        onTextChange={setInputText}
        onAnalyze={() => analyze()}
        onClear={reset}
        loading={loading}
        maxLength={maxInputLength}
      />

      {/* Loading Feedback State */}
      {loading && (
        <LoadingState
          message="Preprocessing tokens and running LinearSVC inference..."
          size={36}
        />
      )}

      {/* Error Message Display */}
      {error && !loading && (
        <ErrorMessage
          message={error}
          onRetry={() => analyze()}
          title="Prediction Request Failed"
        />
      )}

      {/* Prediction Result Display */}
      {prediction && !loading && (
        <div ref={resultRef}>
          <PredictionResult
            prediction={prediction}
            isStale={isStale}
            onReset={reset}
            onReanalyze={() => analyze()}
          />
        </div>
      )}
    </div>
  );
}

export default Detector;
