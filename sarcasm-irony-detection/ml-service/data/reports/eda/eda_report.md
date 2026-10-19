# Phase 2 — EDA Report

**Project**: Sarcasm & Irony Detection System
**Generated**: 2026-08-15 00:16:31

---

## 1. Dataset Overview

| Dataset | Records | Columns |
|---------|--------:|--------:|
| Train   | 81,359 | 3 |
| Test    | 5,527 | 3 |
| **Total** | **86,886** | **3** |

Columns: `text`, `label`, `label_id`

Label mapping: Regular=0, Sarcasm=1, Irony=2, Figurative=3

## 2. Data Quality

### Train Set
- Status: **CLEAN**
- Missing text: 0
- Missing labels: 0
- Empty strings: 0
- Duplicate rows: 0
- Duplicate texts: 13362
- Invalid labels: 0

### Test Set
- Status: **CLEAN**
- Missing text: 0
- Missing labels: 0
- Empty strings: 0
- Duplicate rows: 0
- Duplicate texts: 133
- Invalid labels: 0

## 3. Class Distribution

### Training Set

| Class | Count | Percentage |
|-------|------:|-----------:|
| Regular | 18,569 | 22.82% |
| Sarcasm | 20,678 | 25.42% |
| Irony | 20,877 | 25.66% |
| Figurative | 21,235 | 26.10% |

### Test Set

| Class | Count | Percentage |
|-------|------:|-----------:|
| Regular | 1,858 | 33.62% |
| Sarcasm | 1,602 | 28.98% |
| Irony | 1,301 | 23.54% |
| Figurative | 766 | 13.86% |

### Class Balance Analysis

- **Train**: approximately balanced
  - Majority/minority ratio: 1.14
  - Accuracy is a useful metric, but Macro F1 should still be reported to ensure all classes are evaluated fairly.
- **Test**: heavily imbalanced
  - Majority/minority ratio: 2.43

![Class Distribution](../../results/eda/class_distribution.png)

## 4. Text Length Statistics

| Metric     | Class      |   Count |   Min |   Max |   Mean |   Median |   Std |
|:-----------|:-----------|--------:|------:|------:|-------:|---------:|------:|
| Characters | Overall    |   81359 |     6 |   161 | 102.45 |      107 | 30.71 |
| Characters | Regular    |   18569 |    19 |   156 | 111.85 |      120 | 26.78 |
| Characters | Sarcasm    |   20678 |     8 |   161 |  95.56 |       96 | 31.38 |
| Characters | Irony      |   20877 |     6 |   152 | 103.89 |      109 | 30.5  |
| Characters | Figurative |   21235 |     6 |   161 |  99.53 |      103 | 31.27 |
| Words      | Overall    |   81359 |     1 |    35 |  15.15 |       15 |  5.36 |
| Words      | Regular    |   18569 |     3 |    35 |  14.34 |       14 |  4.44 |
| Words      | Sarcasm    |   20678 |     1 |    32 |  14.65 |       14 |  5.54 |
| Words      | Irony      |   20877 |     1 |    34 |  16.15 |       16 |  5.52 |
| Words      | Figurative |   21235 |     1 |    34 |  15.38 |       15 |  5.59 |

![Character Count Distribution](../../results/eda/character_count_distribution.png)

![Word Count Distribution](../../results/eda/word_count_distribution.png)

![Character Count by Class](../../results/eda/character_count_by_class.png)

![Word Count by Class](../../results/eda/word_count_by_class.png)

## 5. Outlier Analysis

### character_count
- IQR bounds: [-3, 213]
- Low outliers: 0
- High outliers: 0
- Outlier percentage: 0.0%
- **Note**: Outliers are NOT removed. Sarcasm detection may involve intentionally unusual writing.

### word_count
- IQR bounds: [-1, 31]
- Low outliers: 0
- High outliers: 14
- Outlier percentage: 0.02%
- **Note**: Outliers are NOT removed. Sarcasm detection may involve intentionally unusual writing.

### punctuation_count
- IQR bounds: [-5, 19]
- Low outliers: 0
- High outliers: 1261
- Outlier percentage: 1.55%
- **Note**: Outliers are NOT removed. Sarcasm detection may involve intentionally unusual writing.

## 6. Social Media Patterns

### Class-Wise Comparison

| Class      |   Record Count |   Percentage (%) |   Mean Words |   Median Words |   Mean Chars |   Median Chars |   URL % |   Mention % |   Hashtag % |   Emoji % |   Exclamation % |   Question % |   Uppercase Ratio |   AllCaps Words % |   Repeated Chars % |
|:-----------|---------------:|-----------------:|-------------:|---------------:|-------------:|---------------:|--------:|------------:|------------:|----------:|----------------:|-------------:|------------------:|------------------:|-------------------:|
| Regular    |          18569 |            22.82 |         14.3 |             14 |        111.9 |            120 |    79.2 |        18.8 |         100 |       6.2 |            11.1 |          8   |            0.1346 |              29.5 |               14.6 |
| Sarcasm    |          20678 |            25.42 |         14.6 |             14 |         95.6 |             96 |    23.7 |        40.6 |         100 |       8.3 |            20.5 |         14.9 |            0.0813 |              20.1 |               17.7 |
| Irony      |          20877 |            25.66 |         16.1 |             16 |        103.9 |            109 |    27.3 |        35.4 |         100 |       5.7 |            11.9 |         12.3 |            0.0792 |              21.4 |               14   |
| Figurative |          21235 |            26.1  |         15.4 |             15 |         99.5 |            103 |    25.1 |        38.1 |         100 |       6.7 |            16.1 |         13.8 |            0.08   |              21.1 |               15.9 |

![Hashtags by Class](../../results/eda/hashtags_by_class.png)

![Mentions by Class](../../results/eda/mentions_by_class.png)

![Emojis by Class](../../results/eda/emojis_by_class.png)

![Punctuation by Class](../../results/eda/punctuation_by_class.png)

![Uppercase by Class](../../results/eda/uppercase_by_class.png)

### Top Hashtags

| Hashtag | Count |
|---------|------:|
| #sarcasm | 29,733 |
| #irony | 23,704 |
| #ironic | 7,868 |
| #news | 2,770 |
| #humor | 2,757 |
| #peace | 2,705 |
| #education | 2,640 |
| #politics | 2,541 |
| #late | 2,539 |
| #drugs | 2,425 |
| #gopdebate | 1,785 |
| #sarcastic | 1,680 |
| #funny | 1,473 |
| #lol | 923 |
| #love | 623 |

## 7. Word Frequency Analysis

### Top 20 Unigrams (After Stopword Removal)

| Word | Frequency |
|------|----------:|
| #sarcasm | 29,733 |
| #irony | 23,704 |
| #ironic | 7,868 |
| amp | 3,482 |
| like | 3,336 |
| people | 3,236 |
| #news | 2,770 |
| get | 2,758 |
| #humor | 2,757 |
| love | 2,732 |
| #peace | 2,705 |
| #education | 2,640 |
| one | 2,552 |
| #politics | 2,541 |
| #late | 2,539 |
| day | 2,501 |
| #drugs | 2,425 |
| good | 2,226 |
| know | 2,029 |
| time | 1,951 |

![Top Unigrams](../../results/eda/top_unigrams.png)

### Top 20 Bigrams

| Bigram | Frequency |
|--------|----------:|
| #funny #sarcasm | 475 |
| #education #inspirational | 432 |
| #inspirational #lol | 430 |
| #design #tech | 386 |
| #education #design | 365 |
| #tech #long | 365 |
| right #sarcasm | 320 |
| #news #politics | 293 |
| social media | 259 |
| day #sarcasm | 244 |
| fun #sarcasm | 229 |
| #long reads | 226 |
| last night | 225 |
| looks like | 218 |
| donald trump | 216 |
| lol #irony | 185 |
| #peace #love | 183 |
| lol #sarcasm | 182 |
| #funnypictures #funnytweets | 179 |
| #funnytweets #humor | 179 |

![Top Bigrams](../../results/eda/top_bigrams.png)

### Top 20 Trigrams

| Trigram | Frequency |
|---------|----------:|
| #education #inspirational #lol | 430 |
| #design #tech #long | 365 |
| #education #design #tech | 363 |
| #tech #long reads | 226 |
| #funnypictures #funnytweets #humor | 179 |
| pict #funnypictures #funnytweets | 158 |
| internet browsers list | 146 |
| browsers list #internet | 146 |
| list #internet #education | 146 |
| #internet #education #inspirational | 145 |
| #writing #education #inspirational | 112 |
| #news #politics #health | 110 |
| gt #tinder #fails | 109 |
| #tinder #fails #humor | 109 |
| #news #politics #intelmm | 107 |
| #politics #intelmm #osint | 107 |
| #pain #painkillers #drugs | 105 |
| #inspirational #lol #funny | 104 |
| 10 types men | 99 |
| types men would | 98 |

![Top Trigrams](../../results/eda/top_trigrams.png)

## 8. TF-IDF Class-Distinctive Terms (EDA Only)

### Regular
| Term | TF-IDF Score |
|------|------------:|
| late | 0.0454 |
| peace | 0.0428 |
| humor | 0.0379 |
| drugs | 0.0369 |
| education | 0.0369 |
| news | 0.0358 |
| politics | 0.0333 |
| gopdebate | 0.0220 |
| funny | 0.0125 |
| amp | 0.0113 |

### Sarcasm
| Term | TF-IDF Score |
|------|------------:|
| sarcasm | 0.1416 |
| sarcastic | 0.0200 |
| just | 0.0179 |
| love | 0.0173 |
| great | 0.0141 |
| like | 0.0138 |
| good | 0.0134 |
| oh | 0.0117 |
| know | 0.0113 |
| people | 0.0113 |

### Irony
| Term | TF-IDF Score |
|------|------------:|
| irony | 0.1151 |
| ironic | 0.0562 |
| just | 0.0146 |
| people | 0.0130 |
| amp | 0.0114 |
| like | 0.0107 |
| day | 0.0086 |
| don | 0.0084 |
| lol | 0.0076 |
| today | 0.0069 |

### Figurative
| Term | TF-IDF Score |
|------|------------:|
| sarcasm | 0.0710 |
| irony | 0.0562 |
| ironic | 0.0301 |
| just | 0.0156 |
| like | 0.0120 |
| people | 0.0117 |
| love | 0.0115 |
| don | 0.0100 |
| good | 0.0097 |
| day | 0.0094 |

## 9. Vocabulary Analysis

- Total vocabulary: 71,805 unique words
- Total word occurrences: 1,099,917
- Rare words (≤2 occurrences): 49,351 (68.73%)
- Common vocabulary (in all classes): 8,865

| Class | Vocabulary Size | Unique Words |
|-------|----------------:|-------------:|
| Regular | 33,169 | 16,840 |
| Sarcasm | 28,946 | 8,992 |
| Irony | 33,875 | 9,082 |
| Figurative | 32,634 | 4,886 |

## 10. Word Clouds

![Overall Word Cloud](../../results/eda/wordcloud_overall.png)

![Regular Word Cloud](../../results/eda/wordcloud_regular.png)

![Sarcasm Word Cloud](../../results/eda/wordcloud_sarcasm.png)

![Irony Word Cloud](../../results/eda/wordcloud_irony.png)

![Figurative Word Cloud](../../results/eda/wordcloud_figurative.png)

## 11. Important Research Observations

- **Positive words in sarcasm**: Sarcasm class contains positive-looking words: {'love': 1408, 'great': 947, 'wonderful': 104, 'amazing': 110, 'best': 420}
  - *Implication*: Confirms that sarcasm uses verbal irony (positive wording, negative intent)

- **Exaggeration in Sarcasm**: 'so ' appears in 2,118 Sarcasm tweets
  - *Implication*: Exaggeration may serve as a sarcasm/irony signal

- **Exaggeration in Irony**: 'so ' appears in 1,436 Irony tweets
  - *Implication*: Exaggeration may serve as a sarcasm/irony signal

## 12. Data Limitations

- Tweet text only — no metadata (timestamp, user info, conversation context)
- Sarcasm/irony labels are subjective and may have inter-annotator disagreement
- Dataset contains duplicate texts with different labels (text ambiguity)
- Test set distribution differs from training due to overlap removal in Phase 1
- Character limit of ~160 chars constrains linguistic analysis

## 13. Research Questions for Phase 3+

1. Does punctuation preservation improve sarcasm classification?
2. Do hashtags (#sarcasm, #irony) provide useful signal or introduce bias?
3. Does emoji preservation help differentiate classes?
4. Does lemmatization improve TF-IDF performance?
5. Are sarcasm and irony text lengths meaningfully different?
6. Does adding bigrams improve classification accuracy?
7. Does social-media feature engineering improve classical ML models?
8. Can uppercase ratio serve as a distinguishing feature?
9. Does class weighting improve Macro F1 on the imbalanced test set?
10. How do repeated character patterns affect model performance?

## 14. Preprocessing Implications

See `preprocessing_recommendations.md` for detailed recommendations.

---

*This report is generated from Phase 2 EDA. No models were trained.*
*All statistics are from the actual cleaned dataset.*