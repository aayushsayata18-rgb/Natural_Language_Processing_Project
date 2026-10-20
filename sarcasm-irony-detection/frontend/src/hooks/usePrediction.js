/**
 * Custom Hook: usePrediction
 * Manages prediction lifecycle, input state, stale result detection,
 * validation, and race-condition-safe API dispatch.
 */

import { useState, useRef, useCallback } from 'react';
import { predictText } from '../services/api';

export const MAX_INPUT_LENGTH = 5000;

export function usePrediction() {
  const [inputText, setInputText] = useState('');
  const [lastAnalyzedText, setLastAnalyzedText] = useState('');
  const [selectedModel, setSelectedModel] = useState('bow_linearsvm');
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Reference to abort controller for in-flight requests (race condition guard)
  const abortControllerRef = useRef(null);

  /**
   * Handle text changes with automatic stale state tracking
   */
  const handleTextChange = useCallback((text) => {
    setInputText(text);
    if (error) {
      setError(null);
    }
  }, [error]);

  /**
   * Submit text for sarcasm/irony classification
   */
  const analyze = useCallback(async (customText, customModel) => {
    const textToAnalyze = (customText !== undefined ? customText : inputText).trim();
    const modelToUse = customModel || selectedModel || 'bow_linearsvm';

    // Frontend validation
    if (!textToAnalyze) {
      setError('Please enter some text to analyze.');
      return;
    }

    if (textToAnalyze.length > MAX_INPUT_LENGTH) {
      setError(`Text exceeds maximum allowed length of ${MAX_INPUT_LENGTH} characters.`);
      return;
    }

    // Cancel any previous pending request (Section 78 & 79)
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    const controller = new AbortController();
    abortControllerRef.current = controller;

    setLoading(true);
    setError(null);

    try {
      const result = await predictText(textToAnalyze, modelToUse, controller.signal);
      setPrediction(result);
      setLastAnalyzedText(textToAnalyze);
      setLoading(false);
      return result;
    } catch (err) {
      if (err.isCanceled) {
        // Ignored as superseded by newer request
        return;
      }
      setError(err.message || 'Unable to analyze text. Please try again.');
      setLoading(false);
    }
  }, [inputText, selectedModel]);

  /**
   * Reset the analyzer state
   */
  const reset = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    setInputText('');
    setLastAnalyzedText('');
    setPrediction(null);
    setError(null);
    setLoading(false);
  }, []);

  // Is stale if user has typed something different since last prediction
  const isStale = Boolean(prediction && inputText.trim() !== lastAnalyzedText.trim());

  return {
    inputText,
    setInputText: handleTextChange,
    selectedModel,
    setSelectedModel,
    prediction,
    loading,
    error,
    isStale,
    lastAnalyzedText,
    analyze,
    reset,
    maxInputLength: MAX_INPUT_LENGTH,
  };
}

export default usePrediction;

