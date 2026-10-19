# Preprocessing Recommendations for Phase 3

Based on Phase 2 EDA findings. These are recommendations to TEST,
not final decisions.

## 1. Hashtags
- 99.98% of tweets contain hashtags
- Top hashtags: #sarcasm, #irony, #ironic, #news, #humor
- **Recommendation**: PRESERVE hashtags initially. Some hashtags (e.g., #sarcasm, #irony)
  may carry signal. Test with and without hashtag preservation.

## 2. @Mentions
- 33.65% of tweets contain mentions
- **Recommendation**: Replace @mentions with a placeholder token (e.g., @USER)
  to normalize identity references while preserving the mention pattern.

## 3. URLs
- 37.64% of tweets contain URLs
- **Recommendation**: Replace URLs with a placeholder token (e.g., URL)
  or remove entirely. URL content is not accessible.

## 4. Emojis
- 6.72% of tweets contain emojis
- **Recommendation**: Test emoji preservation vs. removal. Emojis may
  carry pragmatic meaning relevant to sarcasm/irony.

## 5. Punctuation
- Exclamation marks and repeated punctuation appear across classes.
- **Recommendation**: Preserve punctuation counts as features. Test whether
  punctuation patterns improve classification.

## 6. Capitalization
- Average uppercase ratio: 0.0926
- 22.84% of tweets have all-caps words
- **Recommendation**: Test case preservation vs. lowercasing.
  Extract uppercase ratio as a feature before lowercasing.

## 7. Repeated Characters
- 15.59% of tweets have repeated chars
- **Recommendation**: Normalize repeated characters (e.g., 'soooo' → 'so')
  but extract a repeated-char count feature first.

## 8. Stopword Removal
- **Recommendation**: Test with and without stopword removal.
  Stopwords may carry contextual meaning in short tweets.

## 9. Lemmatization vs. Stemming
- **Recommendation**: Test lemmatization (preserves word meaning)
  vs. stemming (more aggressive) vs. no morphological normalization.

## 10. N-grams
- Bigrams and trigrams show distinctive patterns per class.
- **Recommendation**: Include bigrams in TF-IDF features.
  Test trigrams if computational budget allows.

## 11. Vocabulary
- Total vocabulary: 71,805 unique words
- Rare words: 49,351 (68.73%)
- **Recommendation**: Apply minimum document frequency (min_df) to filter
  rare words. Test min_df values of 2, 5, and 10.

## 12. Class Balance
- Train balance: approximately balanced
- Test balance: heavily imbalanced
- **Recommendation**: Report Macro F1 as primary metric. For the test set,
  consider stratified evaluation. Test class weighting during training.

---

*These recommendations are based on EDA observations only.*
*Phase 3 should systematically test each decision.*