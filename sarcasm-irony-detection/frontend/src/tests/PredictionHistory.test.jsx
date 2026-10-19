import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import React from 'react';
import PredictionHistory from '../components/PredictionHistory';

describe('PredictionHistory Component', () => {
  const mockPredictions = [
    {
      id: '6713c230e527dca963df991b',
      text: 'What a brilliant meeting on Friday.',
      predictedClass: 'Sarcasm',
      confidence: 0.85,
      modelVersion: 'SIC-v1.0',
      createdAt: '2026-10-19T15:30:00.000Z',
    },
    {
      id: '6713c230e527dca963df991c',
      text: 'Clear blue skies today.',
      predictedClass: 'Regular',
      confidence: 0.92,
      modelVersion: 'SIC-v1.0',
      createdAt: '2026-10-19T15:35:00.000Z',
    },
  ];

  it('renders history table rows with truncated text and badges', () => {
    render(
      <BrowserRouter>
        <PredictionHistory
          predictions={mockPredictions}
          pagination={{ total: 2, page: 1, limit: 20, pages: 1 }}
          loading={false}
          error={null}
          onPageChange={vi.fn()}
          onFilterChange={vi.fn()}
          onDelete={vi.fn()}
        />
      </BrowserRouter>
    );

    expect(screen.getByText('What a brilliant meeting on Friday.')).toBeInTheDocument();
    expect(screen.getByText('Clear blue skies today.')).toBeInTheDocument();
    expect(screen.getByText('85.0%')).toBeInTheDocument();
    expect(screen.getByText('92.0%')).toBeInTheDocument();
    expect(screen.getByText('Total Stored:')).toBeInTheDocument();
  });

  it('renders empty state when prediction list is empty', () => {
    render(
      <BrowserRouter>
        <PredictionHistory
          predictions={[]}
          pagination={{ total: 0, page: 1, limit: 20, pages: 1 }}
          loading={false}
          error={null}
        />
      </BrowserRouter>
    );

    expect(screen.getByText(/No predictions yet/i)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Analyze your first text/i })).toBeInTheDocument();
  });
});
