import { describe, it, expect } from 'vitest';
import {
  formatPercentage,
  formatDateTime,
  truncateText,
  getClassDetails,
} from '../utils/formatting';

describe('Formatting Utilities', () => {
  it('formats decimals and 0-1 probabilities correctly to percentage strings', () => {
    expect(formatPercentage(0.8742)).toBe('87.4%');
    expect(formatPercentage(0.052, 2)).toBe('5.20%');
    expect(formatPercentage(1.0)).toBe('100.0%');
    expect(formatPercentage(0)).toBe('0.0%');
    expect(formatPercentage(null)).toBe('0.0%');
    expect(formatPercentage('invalid')).toBe('0.0%');
  });

  it('formats dates properly', () => {
    const iso = '2026-10-19T15:30:00.000Z';
    const formatted = formatDateTime(iso);
    expect(formatted).not.toBe('—');
    expect(formatDateTime(null)).toBe('—');
    expect(formatDateTime('invalid-date')).toBe('—');
  });

  it('truncates long text properly with ellipsis', () => {
    const text = 'This is a relatively long sentence that needs to be truncated for display in the table.';
    expect(truncateText(text, 20)).toBe('This is a relatively...');
    expect(truncateText('Short text', 50)).toBe('Short text');
    expect(truncateText('')).toBe('');
    expect(truncateText(null)).toBe('');
  });

  it('retrieves class details for all 4 categories', () => {
    const regular = getClassDetails('Regular');
    expect(regular.hex).toBe('#10b981');
    expect(regular.name).toBe('Regular');

    const sarcasm = getClassDetails('Sarcasm');
    expect(sarcasm.hex).toBe('#f59e0b');

    const irony = getClassDetails('Irony');
    expect(irony.hex).toBe('#a855f7');

    const figurative = getClassDetails('Figurative');
    expect(figurative.hex).toBe('#ec4899');

    const unknown = getClassDetails('NonExistent');
    expect(unknown.name).toBe('NonExistent');
  });
});
