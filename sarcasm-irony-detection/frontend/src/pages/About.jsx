import React, { useState, useEffect } from 'react';
import { getModelInfo } from '../services/api';
import {
  Brain,
  Layers,
  Database,
  Cpu,
  CheckCircle2,
  BookOpen,
  Info,
  ShieldAlert,
  BarChart3,
  GitBranch,
  ExternalLink,
} from 'lucide-react';
import { CLASS_COLORS } from '../utils/formatting';

export function About() {
  const [modelInfo, setModelInfo] = useState(null);
  const [loadingModelInfo, setLoadingModelInfo] = useState(true);

  useEffect(() => {
    let isMounted = true;
    const fetchInfo = async () => {
      try {
        const info = await getModelInfo();
        if (isMounted) {
          setModelInfo(info);
          setLoadingModelInfo(false);
        }
      } catch {
        if (isMounted) {
          setLoadingModelInfo(false);
        }
      }
    };
    fetchInfo();
    return () => {
      isMounted = false;
    };
  }, []);

  return (
    <div
      className="about-page"
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '3rem',
        maxWidth: '920px',
        margin: '0 auto',
      }}
    >
      {/* 1. Header */}
      <div>
        <h1
          style={{
            fontSize: '2.4rem',
            color: '#ffffff',
            marginBottom: '0.75rem',
            fontFamily: 'var(--font-heading)',
          }}
        >
          About the Project
        </h1>
        <p style={{ color: '#94a3b8', fontSize: '1.05rem', lineHeight: 1.6 }}>
          Comprehensive documentation of the Sarcasm and Irony Detection System developed for the
          NLP Project curriculum.
        </p>
      </div>

      {/* 2. Problem Statement & Motivation */}
      <section className="card" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <h2 style={{ fontSize: '1.35rem', color: '#ffffff', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <BookOpen size={20} style={{ color: '#6366f1' }} />
          Problem Statement & Academic Motivation
        </h2>
        <p style={{ color: '#cbd5e1', fontSize: '0.95rem', lineHeight: 1.7 }}>
          Sarcasm and irony present formidable challenges in Natural Language Processing because
          the literal sentiment of the tokens often stands in direct opposition to the speaker's true
          intended message. A sentence such as <em>"What wonderful weather for a marathon"</em> spoken
          during a severe storm contains positive lexical terms but expresses frustration.
        </p>
        <p style={{ color: '#cbd5e1', fontSize: '0.95rem', lineHeight: 1.7 }}>
          Traditional sentiment analysis models frequently misclassify sarcastic statements as positive.
          The primary objective of this project is to develop, evaluate, package, and deploy a robust
          four-class classification model capable of distinguishing between regular statements, sarcasm,
          irony, and other figurative expressions.
        </p>
      </section>

      {/* 3. Cross-Paradigm Benchmark Matrix (Section 39 & 40) */}
      <section className="card" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
          <h2 style={{ fontSize: '1.35rem', color: '#ffffff', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <BarChart3 size={20} style={{ color: '#10b981' }} />
            Cross-Paradigm Model Benchmark Matrix
          </h2>
          <span
            style={{
              fontSize: '0.75rem',
              fontFamily: 'var(--font-mono)',
              color: '#38bdf8',
              background: 'rgba(56, 189, 248, 0.1)',
              padding: '0.2rem 0.6rem',
              borderRadius: 'var(--radius-full)',
              border: '1px solid rgba(56, 189, 248, 0.2)',
            }}
          >
            5 Architectures Benchmarked & Deployed
          </span>
        </div>

        <p style={{ color: '#94a3b8', fontSize: '0.9rem', lineHeight: 1.6 }}>
          All five models were evaluated on the identical decontaminated benchmark of 5,527 test tweets
          (Regular: 33.6%, Sarcasm: 29.0%, Irony: 23.5%, Figurative: 13.9%). All 5 models are hosted live
          and selectable in the detector dropdown.
        </p>

        {/* Responsive Table */}
        <div style={{ overflowX: 'auto', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem', textAlign: 'left' }}>
            <thead>
              <tr style={{ background: 'rgba(255, 255, 255, 0.04)', borderBottom: '1px solid var(--border-color)' }}>
                <th style={{ padding: '0.75rem 1rem', color: '#f8fafc' }}>Architecture</th>
                <th style={{ padding: '0.75rem 0.75rem', color: '#f8fafc' }}>Representation</th>
                <th style={{ padding: '0.75rem 0.75rem', color: '#f8fafc' }}>Parameters</th>
                <th style={{ padding: '0.75rem 0.75rem', color: '#f8fafc' }}>Latency</th>
                <th style={{ padding: '0.75rem 0.75rem', color: '#f8fafc' }}>Accuracy</th>
                <th style={{ padding: '0.75rem 0.75rem', color: '#f8fafc' }}>Macro F1</th>
                <th style={{ padding: '0.75rem 0.75rem', color: '#f8fafc' }}>Figurative F1</th>
                <th style={{ padding: '0.75rem 1rem', color: '#f8fafc' }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {[
                {
                  arch: 'LinearSVM (BoW)',
                  rep: 'Bag of Words (n-grams 1-2)',
                  params: '112,312',
                  latency: '0.027 ms',
                  acc: '84.57%',
                  f1: '0.7187',
                  figF1: '0.1378',
                  status: '⭐ Selected Champion',
                  isChamp: true,
                },
                {
                  arch: 'LinearSVM (TF-IDF)',
                  rep: 'Sublinear TF-IDF',
                  params: '112,312',
                  latency: '0.014 ms',
                  acc: '85.65%',
                  f1: '0.7002',
                  figF1: '0.0421',
                  status: 'Live in Dropdown',
                  isChamp: false,
                },
                {
                  arch: 'GloVe + Bi-LSTM',
                  rep: 'Pretrained 100d GloVe',
                  params: '3,197,636',
                  latency: '0.041 ms',
                  acc: '86.12%',
                  f1: '0.6940',
                  figF1: '0.0026',
                  status: 'Live in Dropdown',
                  isChamp: false,
                },
                {
                  arch: 'FastText + Bi-LSTM + Attention',
                  rep: '300d Subwords + Attention',
                  params: '9,204,828',
                  latency: '0.098 ms',
                  acc: '86.03%',
                  f1: '0.6928',
                  figF1: '0.0000',
                  status: 'Live in Dropdown',
                  isChamp: false,
                },
                {
                  arch: 'BERT Contextual',
                  rep: '12-Layer Transformer (768d)',
                  params: '109,485,316',
                  latency: '39.33 ms',
                  acc: '86.12%',
                  f1: '0.6935',
                  figF1: '0.0000',
                  status: 'Live in Dropdown',
                  isChamp: false,
                },
              ].map((row, idx) => (
                <tr
                  key={idx}
                  style={{
                    borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
                    background: row.isChamp ? 'rgba(16, 185, 129, 0.06)' : 'transparent',
                  }}
                >
                  <td style={{ padding: '0.75rem 1rem', fontWeight: 600, color: row.isChamp ? '#10b981' : '#f8fafc' }}>
                    {row.arch}
                  </td>
                  <td style={{ padding: '0.75rem 0.75rem', color: '#cbd5e1' }}>{row.rep}</td>
                  <td style={{ padding: '0.75rem 0.75rem', color: '#94a3b8', fontFamily: 'var(--font-mono)' }}>
                    {row.params}
                  </td>
                  <td style={{ padding: '0.75rem 0.75rem', color: '#38bdf8', fontFamily: 'var(--font-mono)' }}>
                    {row.latency}
                  </td>
                  <td style={{ padding: '0.75rem 0.75rem', color: '#cbd5e1', fontFamily: 'var(--font-mono)' }}>
                    {row.acc}
                  </td>
                  <td style={{ padding: '0.75rem 0.75rem', fontWeight: 700, color: row.isChamp ? '#10b981' : '#cbd5e1', fontFamily: 'var(--font-mono)' }}>
                    {row.f1}
                  </td>
                  <td style={{ padding: '0.75rem 0.75rem', color: row.isChamp ? '#10b981' : '#ef4444', fontFamily: 'var(--font-mono)' }}>
                    {row.figF1}
                  </td>
                  <td style={{ padding: '0.75rem 1rem' }}>
                    <span
                      style={{
                        fontSize: '0.725rem',
                        padding: '0.15rem 0.5rem',
                        borderRadius: 'var(--radius-full)',
                        background: row.isChamp ? 'rgba(16, 185, 129, 0.15)' : 'rgba(255, 255, 255, 0.05)',
                        color: row.isChamp ? '#10b981' : '#94a3b8',
                        border: `1px solid ${row.isChamp ? 'rgba(16, 185, 129, 0.3)' : 'var(--border-color)'}`,
                        fontWeight: 600,
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {row.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Academic Key Takeaways Callout */}
        <div
          style={{
            background: 'rgba(99, 102, 241, 0.05)',
            border: '1px solid rgba(99, 102, 241, 0.25)',
            borderRadius: '8px',
            padding: '1.25rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.65rem',
          }}
        >
          <h4 style={{ color: '#818cf8', margin: 0, fontSize: '0.95rem' }}>
            Academic Research Insights: Why Did LinearSVM Win on Macro F1?
          </h4>
          <ul style={{ margin: 0, paddingLeft: '1.2rem', color: '#cbd5e1', fontSize: '0.85rem', lineHeight: 1.6 }}>
            <li>
              <strong>Margin Maximization vs. Cross-Entropy Loss:</strong> LinearSVM optimizes geometric margins between hyperplanes. In our imbalanced dataset, standard unweighted cross-entropy in deep neural networks flattened the minority <em>Figurative</em> class into the majority classes, whereas LinearSVM preserved 16.97% recall.
            </li>
            <li>
              <strong>BERT's Semantic Precision:</strong> While BERT collapsed on Figurative, it achieved <strong>0 errors out of 2,903 tweets</strong> between Sarcasm and Irony, proving contextual transformers have superior phrase-level boundary precision.
            </li>
            <li>
              <strong>Computational Return on Investment:</strong> LinearSVM executes at <strong>0.027 ms/text</strong> (~1,400× faster than BERT on CPU) with 975× fewer parameters, making it optimal for high-throughput production deployment.
            </li>
          </ul>
        </div>
      </section>

      {/* 4. Dataset Information (Section 42) */}
      <section className="card" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.75rem' }}>
          <h2 style={{ fontSize: '1.35rem', color: '#ffffff', display: 'flex', alignItems: 'center', gap: '0.5rem', margin: 0 }}>
            <Database size={20} style={{ color: '#f59e0b' }} />
            Dataset & Corpus Details
          </h2>
          <a
            href="https://www.kaggle.com/datasets/nikhiljohnk/tweets-with-sarcasm-and-irony"
            target="_blank"
            rel="noopener noreferrer"
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.4rem',
              fontSize: '0.8rem',
              fontWeight: 500,
              color: '#38bdf8',
              background: 'rgba(56, 189, 248, 0.1)',
              padding: '0.35rem 0.75rem',
              borderRadius: 'var(--radius-full)',
              border: '1px solid rgba(56, 189, 248, 0.25)',
              textDecoration: 'none',
              transition: 'all 0.2s ease',
            }}
          >
            <span>Kaggle: Tweets with Sarcasm and Irony</span>
            <ExternalLink size={14} />
          </a>
        </div>

        <p style={{ color: '#cbd5e1', fontSize: '0.95rem', lineHeight: 1.7, margin: 0 }}>
          The project utilizes the <strong>Tweets with Sarcasm and Irony</strong> dataset sourced from Kaggle
          (curated by <strong>Nikhil John K</strong>), capturing real-world social media discourse with rich pragmatic nuance.
          The dataset provides supervised labels across four distinct linguistic categories, grounded in the standardized
          SemEval-2018 Task 3 benchmark for figurative language understanding.
        </p>

        {/* Dataset Key Volume Cards */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))',
            gap: '0.75rem',
          }}
        >
          <div style={{ padding: '0.85rem 1rem', background: 'rgba(255, 255, 255, 0.03)', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
            <span style={{ fontSize: '0.75rem', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Raw Corpus</span>
            <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#f8fafc', marginTop: '0.2rem' }}>89,536 Tweets</div>
            <span style={{ fontSize: '0.75rem', color: '#64748b' }}>81,408 Train / 8,128 Test</span>
          </div>

          <div style={{ padding: '0.85rem 1rem', background: 'rgba(255, 255, 255, 0.03)', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
            <span style={{ fontSize: '0.75rem', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Clean Training Set</span>
            <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#10b981', marginTop: '0.2rem' }}>81,359 Tweets</div>
            <span style={{ fontSize: '0.75rem', color: '#64748b' }}>Deduplicated & leak-free</span>
          </div>

          <div style={{ padding: '0.85rem 1rem', background: 'rgba(255, 255, 255, 0.03)', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
            <span style={{ fontSize: '0.75rem', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Benchmark Test Set</span>
            <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#38bdf8', marginTop: '0.2rem' }}>5,527 Tweets</div>
            <span style={{ fontSize: '0.75rem', color: '#64748b' }}>Strictly decontaminated</span>
          </div>

          <div style={{ padding: '0.85rem 1rem', background: 'rgba(255, 255, 255, 0.03)', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
            <span style={{ fontSize: '0.75rem', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Classification</span>
            <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#a855f7', marginTop: '0.2rem' }}>4 Classes</div>
            <span style={{ fontSize: '0.75rem', color: '#64748b' }}>Pragmatic & Figurative</span>
          </div>
        </div>

        {/* 4-Class Test Breakdown Cards */}
        <div>
          <h4 style={{ fontSize: '0.9rem', color: '#f8fafc', marginBottom: '0.6rem' }}>
            Class Distribution Across Benchmark Evaluation Set (5,527 Tweets):
          </h4>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
              gap: '0.65rem',
            }}
          >
            {[
              {
                name: 'Regular',
                count: '1,858',
                percent: '33.6%',
                color: '#10b981',
                bg: 'rgba(16, 185, 129, 0.08)',
                desc: 'Literal expressions without sarcasm or irony.',
              },
              {
                name: 'Sarcasm',
                count: '1,602',
                percent: '29.0%',
                color: '#f59e0b',
                bg: 'rgba(245, 158, 11, 0.08)',
                desc: 'Mockery where surface praise conveys contempt.',
              },
              {
                name: 'Irony',
                count: '1,301',
                percent: '23.5%',
                color: '#a855f7',
                bg: 'rgba(168, 85, 247, 0.08)',
                desc: 'Incongruity between expectation and actual reality.',
              },
              {
                name: 'Figurative',
                count: '766',
                percent: '13.9%',
                color: '#ec4899',
                bg: 'rgba(236, 72, 153, 0.08)',
                desc: 'Metaphors, hyperbole, and creative non-literal expressions.',
              },
            ].map((cls, idx) => (
              <div
                key={idx}
                style={{
                  padding: '0.75rem 0.9rem',
                  background: cls.bg,
                  borderRadius: '8px',
                  border: `1px solid ${cls.color}33`,
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.25rem',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontWeight: 600, color: cls.color, fontSize: '0.9rem' }}>{cls.name}</span>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: '#cbd5e1' }}>
                    {cls.count} ({cls.percent})
                  </span>
                </div>
                <span style={{ fontSize: '0.75rem', color: '#94a3b8', lineHeight: 1.4 }}>{cls.desc}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Curation & Decontamination Audit */}
        <div style={{ background: 'rgba(255, 255, 255, 0.02)', padding: '1rem', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
          <h4 style={{ color: '#e2e8f0', fontSize: '0.9rem', marginBottom: '0.5rem' }}>
            Data Quality & Leakage Decontamination Protocol:
          </h4>
          <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '0.45rem', padding: 0, margin: 0, color: '#94a3b8', fontSize: '0.85rem' }}>
            <li style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem' }}>
              <CheckCircle2 size={15} style={{ color: '#10b981', marginTop: '2px', flexShrink: 0 }} />
              <span><strong>Cross-Split Leakage Removal:</strong> 2,592 duplicate or label-conflicting samples across train/test splits were purged to ensure 0% test data memorization.</span>
            </li>
            <li style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem' }}>
              <CheckCircle2 size={15} style={{ color: '#10b981', marginTop: '2px', flexShrink: 0 }} />
              <span><strong>Social Media Text Normalization:</strong> Normalized user handles (`@user`), stripped noisy t.co URLs, unpacked contractions, and decoded UTF-8 emojis into descriptive tokens.</span>
            </li>
            <li style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem' }}>
              <CheckCircle2 size={15} style={{ color: '#10b981', marginTop: '2px', flexShrink: 0 }} />
              <span><strong>Minority Class Preservation:</strong> Retained all natural class imbalances to honestly benchmark whether models collapse on rare figurative speech (13.9% support).</span>
            </li>
          </ul>
        </div>
      </section>

      {/* 5. Methodology & NLP Pipeline (Section 38 & 43) */}
      <section className="card" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <h2 style={{ fontSize: '1.35rem', color: '#ffffff', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <GitBranch size={20} style={{ color: '#a855f7' }} />
          NLP Pipeline & 5 Evaluated Paradigms
        </h2>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          {[
            { step: 'Bag of Words (BoW)', detail: 'CountVectorizer unigrams and bigrams (28,077 features) with One-vs-Rest LinearSVC and Platt probability calibration.' },
            { step: 'TF-IDF Representation', detail: 'Sublinear TF-IDF term frequency scaling downweighting common terms and highlighting rare contrastive tokens.' },
            { step: 'GloVe Word Embeddings', detail: 'Pretrained 100-dimensional GloVe dense word embeddings feeding a 2-layer Bidirectional LSTM network.' },
            { step: 'FastText with Self-Attention', detail: 'Character n-gram subword synthesis resolving 93.8% of Twitter OOV tokens, augmented with Additive Self-Attention.' },
            { step: 'BERT Contextual Transformer', detail: 'Pretrained bert-base-uncased contextual transformer (12 layers, 768d, 12 attention heads) fine-tuned end-to-end with AdamW.' },
          ].map((item, idx) => (
            <div
              key={idx}
              style={{
                display: 'flex',
                gap: '0.75rem',
                padding: '0.75rem 1rem',
                borderRadius: '8px',
                background: 'rgba(255, 255, 255, 0.02)',
                border: '1px solid var(--border-color)',
                fontSize: '0.9rem',
              }}
            >
              <span style={{ fontWeight: 600, color: '#f8fafc', whiteSpace: 'nowrap' }}>
                {item.step}:
              </span>
              <span style={{ color: '#94a3b8' }}>{item.detail}</span>
            </div>
          ))}
        </div>
      </section>

      {/* 6. Technology Stack (Section 41) */}
      <section className="card" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <h2 style={{ fontSize: '1.35rem', color: '#ffffff', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Layers size={20} style={{ color: '#38bdf8' }} />
          Complete Technology Stack
        </h2>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
            gap: '1rem',
          }}
        >
          <div style={{ padding: '1rem', background: 'rgba(255, 255, 255, 0.02)', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
            <h4 style={{ color: '#6366f1', marginBottom: '0.35rem' }}>Frontend</h4>
            <p style={{ color: '#94a3b8', fontSize: '0.85rem' }}>React 18, Vite, React Router, Axios, Vanilla CSS Design System</p>
          </div>
          <div style={{ padding: '1rem', background: 'rgba(255, 255, 255, 0.02)', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
            <h4 style={{ color: '#10b981', marginBottom: '0.35rem' }}>Backend API</h4>
            <p style={{ color: '#94a3b8', fontSize: '0.85rem' }}>Node.js, Express, Rate Limiting, Helmet, Morgan, Supertest</p>
          </div>
          <div style={{ padding: '1rem', background: 'rgba(255, 255, 255, 0.02)', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
            <h4 style={{ color: '#38bdf8', marginBottom: '0.35rem' }}>ML Service</h4>
            <p style={{ color: '#94a3b8', fontSize: '0.85rem' }}>Python 3.14, FastAPI, Uvicorn, scikit-learn, joblib, Pydantic</p>
          </div>
          <div style={{ padding: '1rem', background: 'rgba(255, 255, 255, 0.02)', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
            <h4 style={{ color: '#f59e0b', marginBottom: '0.35rem' }}>Database</h4>
            <p style={{ color: '#94a3b8', fontSize: '0.85rem' }}>MongoDB, Mongoose ODM, Prediction History persistence</p>
          </div>
        </div>
      </section>

      {/* 7. Limitations & Ethical Considerations (Section 87 & 88) */}
      <section
        style={{
          background: 'rgba(245, 158, 11, 0.05)',
          border: '1px solid rgba(245, 158, 11, 0.25)',
          borderRadius: 'var(--radius-lg)',
          padding: '1.75rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '0.75rem',
        }}
      >
        <h3 style={{ fontSize: '1.15rem', color: '#fbbf24', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <ShieldAlert size={20} />
          Limitations & Academic Disclaimers
        </h3>
        <p style={{ color: '#cbd5e1', fontSize: '0.9rem', lineHeight: 1.6 }}>
          Language comprehension by machines is probabilistic. Sarcasm often hinges upon unspoken cultural
          context, shared personal history, vocal inflection, and facial expressions that are absent in plain
          isolated text. Consequently, model classifications represent mathematical likelihoods based on
          patterns observed in the training corpus and should not be treated as infallible assessments of
          human intent.
        </p>
      </section>
    </div>
  );
}

export default About;
