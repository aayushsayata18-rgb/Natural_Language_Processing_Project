import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import React from 'react';
import PredictionResult from '../components/PredictionResult';

describe('PredictionResult Component', () => {
  const mockPrediction = {
    text: 'What great news!',
    predicted_class: 'Sarcasm',
    confidence: 0.884,
    probabilities: {
      Regular: 0.04,
      Sarcasm: 0.884,
      Irony: 0.046,
      Figurative: 0.03,
    },
    model_name: 'SarcasmIronyClassifier-LinearSVM',
    model_version: 'SIC-v1.0',
    latency_ms: 1.45,
  };

  it('renders predicted class, confidence, and model metadata', () => {
    render(<PredictionResult prediction={mockPrediction} isStale={false} />);

    expect(screen.getByText(/Detected:/i)).toBeInTheDocument();
    expect(screen.getAllByText('88.4%').length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText(/SarcasmIronyClassifier-LinearSVM/i)).toBeInTheDocument();
    expect(screen.getByText(/1.45 ms/i)).toBeInTheDocument();
    expect(
      screen.getByText(/Sarcasm and irony can be context-dependent and subjective/i)
    ).toBeInTheDocument();
  });

  it('displays stale indicator when isStale is true', () => {
    const handleReanalyze = vi.fn();

    render(
      <PredictionResult
        prediction={mockPrediction}
        isStale={true}
        onReanalyze={handleReanalyze}
      />
    );

    expect(
      screen.getByText(/The text above has been modified since this prediction/i)
    ).toBeInTheDocument();

    const updateBtn = screen.getByRole('button', { name: /Update Result/i });
    fireEvent.click(updateBtn);
    expect(handleReanalyze).toHaveBeenCalledTimes(1);
  });
});
