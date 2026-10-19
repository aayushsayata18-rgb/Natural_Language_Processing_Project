import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import React from 'react';
import ProbabilityChart from '../components/ProbabilityChart';

describe('ProbabilityChart Component', () => {
  it('renders all four classes and their formatted probabilities', () => {
    const mockProbabilities = {
      Regular: 0.12,
      Sarcasm: 0.78,
      Irony: 0.06,
      Figurative: 0.04,
    };

    render(
      <ProbabilityChart
        probabilities={mockProbabilities}
        predictedClass="Sarcasm"
      />
    );

    expect(screen.getByText('Regular (Literal / Non-Sarcastic)')).toBeInTheDocument();
    expect(screen.getByText('Sarcasm')).toBeInTheDocument();
    expect(screen.getByText('Irony')).toBeInTheDocument();
    expect(screen.getByText('Figurative Language')).toBeInTheDocument();

    expect(screen.getByText('78.0%')).toBeInTheDocument();
    expect(screen.getByText('12.0%')).toBeInTheDocument();
    expect(screen.getByText('PREDICTED')).toBeInTheDocument();
  });

  it('renders fallback when probabilities are invalid or malformed', () => {
    render(<ProbabilityChart probabilities={null} predictedClass="Sarcasm" />);
    expect(
      screen.getByText('Prediction details are temporarily unavailable.')
    ).toBeInTheDocument();
  });
});
