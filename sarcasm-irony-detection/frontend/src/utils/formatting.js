/**
 * Formatting and Helper Utilities.
 * Sarcasm & Irony Detection System - React Frontend
 */

export const CLASS_COLORS = {
  Regular: {
    hex: '#10b981',
    bg: 'rgba(16, 185, 129, 0.12)',
    badgeClass: 'badge-regular',
    name: 'Regular',
    label: 'Regular (Literal / Non-Sarcastic)',
    description: 'Direct literal communication where the conveyed meaning matches the explicit surface statement.',
  },
  Sarcasm: {
    hex: '#f59e0b',
    bg: 'rgba(245, 158, 11, 0.12)',
    badgeClass: 'badge-sarcasm',
    name: 'Sarcasm',
    label: 'Sarcasm',
    description: 'Mocking or caustic expression where surface praise conveys disparagement, contempt, or sharp ridicule.',
  },
  Irony: {
    hex: '#a855f7',
    bg: 'rgba(168, 85, 247, 0.12)',
    badgeClass: 'badge-irony',
    name: 'Irony',
    label: 'Irony',
    description: 'Incongruity between expectation and reality, or words conveying the opposite of their literal meaning without malice.',
  },
  Figurative: {
    hex: '#ec4899',
    bg: 'rgba(236, 72, 153, 0.12)',
    badgeClass: 'badge-figurative',
    name: 'Figurative',
    label: 'Figurative Language',
    description: 'Metaphorical, hyperbolic, or non-literal expressions that do not fit into direct sarcasm or situational irony.',
  },
};

/**
 * Format a probability/confidence value (0 to 1) into a clean percentage string.
 * @param {number|string} val
 * @param {number} decimals
 * @returns {string} e.g. "87.4%"
 */
export function formatPercentage(val, decimals = 1) {
  if (val === null || val === undefined || isNaN(Number(val))) {
    return '0.0%';
  }
  let num = Number(val);
  // If already on 0-100 scale, keep; if 0-1 scale, multiply by 100
  if (num > 0 && num <= 1.0) {
    num = num * 100;
  }
  return `${num.toFixed(decimals)}%`;
}

/**
 * Format ISO date string into readable local timestamp.
 * @param {string|Date} isoString
 * @returns {string} e.g. "Oct 19, 2026, 9:45 PM"
 */
export function formatDateTime(isoString) {
  if (!isoString) return '—';
  try {
    const date = new Date(isoString);
    if (isNaN(date.getTime())) return '—';
    return date.toLocaleString(undefined, {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return '—';
  }
}

/**
 * Truncate long text with ellipsis.
 * @param {string} text
 * @param {number} maxLen
 * @returns {string}
 */
export function truncateText(text, maxLen = 80) {
  if (!text) return '';
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen).trim() + '...';
}

/**
 * Retrieve metadata details for a predicted class.
 * @param {string} className
 */
export function getClassDetails(className) {
  return (
    CLASS_COLORS[className] || {
      hex: '#94a3b8',
      bg: 'rgba(148, 163, 184, 0.12)',
      badgeClass: 'badge-regular',
      name: className || 'Unknown',
      label: className || 'Unknown',
      description: 'NLP Classification result.',
    }
  );
}
