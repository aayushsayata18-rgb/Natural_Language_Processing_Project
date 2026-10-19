import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import React from 'react';
import Detector from '../pages/Detector';
import * as api from '../services/api';

vi.mock('../services/api');

describe('Detector Integration Test', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('runs complete flow: text entry -> predictText call -> result rendered', async () => {
    const mockPredictionResponse = {
      text: 'What wonderful weather!',
      predicted_class: 'Sarcasm',
      confidence: 0.892,
      probabilities: {
        Regular: 0.05,
        Sarcasm: 0.892,
        Irony: 0.038,
        Figurative: 0.02,
      },
      model_name: 'SarcasmIronyClassifier-LinearSVM',
      model_version: 'SIC-v1.0',
      latency_ms: 1.15,
      history_saved: true,
    };

    api.predictText.mockResolvedValueOnce(mockPredictionResponse);

    render(<Detector />);

    // Check header
    expect(screen.getByText('Sarcasm & Irony Detector')).toBeInTheDocument();

    // Type text into textarea
    const textarea = screen.getByPlaceholderText(/Type or paste text here/i);
    fireEvent.change(textarea, { target: { value: 'What wonderful weather!' } });

    // Click Analyze button
    const analyzeBtn = screen.getByRole('button', { name: /Analyze Text/i });
    fireEvent.click(analyzeBtn);

    // Verify loading state appears
    expect(screen.getByText(/Preprocessing tokens and running LinearSVC/i)).toBeInTheDocument();

    // Wait for prediction result
    await waitFor(() => {
      expect(screen.getByText(/Detected:/i)).toBeInTheDocument();
      expect(screen.getAllByText('89.2%').length).toBeGreaterThanOrEqual(1);
    });

    expect(api.predictText).toHaveBeenCalledTimes(1);
  });
});
