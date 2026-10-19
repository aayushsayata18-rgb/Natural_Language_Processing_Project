import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import React from 'react';
import TextAnalyzer from '../components/TextAnalyzer';

describe('TextAnalyzer Component', () => {
  it('renders textarea, character counter, and buttons', () => {
    const handleTextChange = vi.fn();
    const handleAnalyze = vi.fn();
    const handleClear = vi.fn();

    render(
      <TextAnalyzer
        inputText="Sample text"
        onTextChange={handleTextChange}
        onAnalyze={handleAnalyze}
        onClear={handleClear}
        loading={false}
        maxLength={5000}
      />
    );

    const textarea = screen.getByPlaceholderText(/Type or paste text here/i);
    expect(textarea).toBeInTheDocument();
    expect(screen.getByText('11 / 5000')).toBeInTheDocument();

    const analyzeBtn = screen.getByRole('button', { name: /Analyze Text/i });
    expect(analyzeBtn).toBeInTheDocument();
    expect(analyzeBtn).not.toBeDisabled();
  });

  it('disables Analyze button when input is empty or loading', () => {
    const { rerender } = render(
      <TextAnalyzer
        inputText="   "
        onTextChange={vi.fn()}
        onAnalyze={vi.fn()}
        onClear={vi.fn()}
        loading={false}
      />
    );

    const analyzeBtn = screen.getByRole('button', { name: /Analyze Text/i });
    expect(analyzeBtn).toBeDisabled();

    rerender(
      <TextAnalyzer
        inputText="Valid text"
        onTextChange={vi.fn()}
        onAnalyze={vi.fn()}
        onClear={vi.fn()}
        loading={true}
      />
    );

    expect(screen.getByRole('button', { name: /Analyzing.../i })).toBeDisabled();
  });

  it('populates textarea when an example pill is clicked', () => {
    const handleTextChange = vi.fn();

    render(
      <TextAnalyzer
        inputText=""
        onTextChange={handleTextChange}
        onAnalyze={handleAnalyze => {}}
        onClear={() => {}}
        loading={false}
      />
    );

    const exampleBtn = screen.getByRole('button', { name: /Meeting/i });
    fireEvent.click(exampleBtn);

    expect(handleTextChange).toHaveBeenCalledWith(
      'Fantastic, another three-hour meeting on a Friday afternoon.'
    );
  });
});
