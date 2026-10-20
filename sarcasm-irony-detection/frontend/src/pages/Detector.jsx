import React, { useRef, useEffect, useState } from 'react';
import usePrediction from '../hooks/usePrediction';
import TextAnalyzer from '../components/TextAnalyzer';
import PredictionResult from '../components/PredictionResult';
import LoadingState from '../components/LoadingState';
import ErrorMessage from '../components/ErrorMessage';
import { getAvailableModels, DEFAULT_MODELS } from '../services/api';
import { Sparkles, Cpu, AlertCircle } from 'lucide-react';

export function Detector() {
  const {
    inputText,
    setInputText,
    selectedModel,
    setSelectedModel,
    prediction,
    loading,
    error,
    isStale,
    analyze,
    reset,
    maxInputLength,
  } = usePrediction();

  const [availableModels, setAvailableModels] = useState(DEFAULT_MODELS);
  const resultRef = useRef(null);

  // Fetch available models list on mount
  useEffect(() => {
    let isMounted = true;
    getAvailableModels().then((models) => {
      if (isMounted && Array.isArray(models) && models.length > 0) {
        setAvailableModels(models);
      }
    }).catch(() => {});
    return () => {
      isMounted = false;
    };
  }, []);

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

  // Dynamic loading feedback message based on active model paradigm
  const getLoadingMessage = () => {
    switch (selectedModel) {
      case 'bert_transformer':
        return 'Encoding WordPiece tokens and running 12-layer BERT Transformer inference (~39ms)...';
      case 'fasttext_attention':
        return 'Synthesizing FastText subwords and running Bi-LSTM with Additive Attention...';
      case 'glove_bilstm':
        return 'Mapping GloVe 100d embeddings and running Bidirectional LSTM sequence inference...';
      case 'tfidf_linearsvm':
        return 'Calculating sublinear TF-IDF n-grams and running LinearSVC decision scoring...';
      default:
        return 'Extracting Bag-of-Words tokens and running LinearSVC champion model inference...';
    }
  };

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
        <p style={{ color: '#94a3b8', fontSize: '1rem', maxWidth: '650px', margin: '0 auto' }}>
          Compare predictions across 5 benchmarked NLP paradigms: BoW, TF-IDF, GloVe Bi-LSTM,
          FastText Attention, and BERT Contextual Transformer.
        </p>
      </div>

      {/* Input Component */}
      <TextAnalyzer
        inputText={inputText}
        onTextChange={setInputText}
        onAnalyze={() => analyze()}
        onClear={reset}
        selectedModel={selectedModel}
        onModelChange={setSelectedModel}
        models={availableModels}
        loading={loading}
        maxLength={maxInputLength}
      />

      {/* Loading Feedback State */}
      {loading && (
        <LoadingState
          message={getLoadingMessage()}
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

