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
          MCA Final-Year NLP curriculum.
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

      {/* 3. Model Information & Performance Metrics (Section 39 & 40) */}
      <section className="card" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
          <h2 style={{ fontSize: '1.35rem', color: '#ffffff', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <BarChart3 size={20} style={{ color: '#10b981' }} />
            Final Production Model Specifications
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
            Phase 9 Selection & Phase 10 Packaged
          </span>
        </div>

        <p style={{ color: '#94a3b8', fontSize: '0.9rem' }}>
          Selected after exhaustive benchmarking against Logistic Regression, Naive Bayes, Random
          Forest, BiLSTM, and Transformer baselines in Phase 9.
        </p>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
            gap: '1rem',
          }}
        >
          <div
            style={{
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid var(--border-color)',
              borderRadius: '10px',
              padding: '1rem',
            }}
          >
            <div style={{ fontSize: '0.75rem', color: '#94a3b8', textTransform: 'uppercase' }}>Model Name</div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.05rem', fontWeight: 600, color: '#f8fafc', marginTop: '0.25rem' }}>
              {modelInfo?.model_name || 'LinearSVM'}
            </div>
          </div>

          <div
            style={{
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid var(--border-color)',
              borderRadius: '10px',
              padding: '1rem',
            }}
          >
            <div style={{ fontSize: '0.75rem', color: '#94a3b8', textTransform: 'uppercase' }}>Model Version</div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.05rem', fontWeight: 600, color: '#6366f1', marginTop: '0.25rem' }}>
              {modelInfo?.model_version || 'SIC-v1.0'}
            </div>
          </div>

          <div
            style={{
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid var(--border-color)',
              borderRadius: '10px',
              padding: '1rem',
            }}
          >
            <div style={{ fontSize: '0.75rem', color: '#94a3b8', textTransform: 'uppercase' }}>Macro F1-Score</div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.05rem', fontWeight: 700, color: '#10b981', marginTop: '0.25rem' }}>
              {modelInfo?.macro_f1 ? modelInfo.macro_f1.toFixed(4) : '0.7719'}
            </div>
          </div>

          <div
            style={{
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid var(--border-color)',
              borderRadius: '10px',
              padding: '1rem',
            }}
          >
            <div style={{ fontSize: '0.75rem', color: '#94a3b8', textTransform: 'uppercase' }}>Overall Accuracy</div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.05rem', fontWeight: 700, color: '#38bdf8', marginTop: '0.25rem' }}>
              {modelInfo?.accuracy ? `${(modelInfo.accuracy * 100).toFixed(2)}%` : '77.00%'}
            </div>
          </div>
        </div>

        <div style={{ fontSize: '0.85rem', color: '#94a3b8', marginTop: '0.25rem' }}>
          <strong>Architecture:</strong> {modelInfo?.model_type || 'LinearSVC (One-vs-Rest) with BoW CountVectorizer (n-grams: 1-2)'}
        </div>
      </section>

      {/* 4. Dataset Information (Section 42) */}
      <section className="card" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <h2 style={{ fontSize: '1.35rem', color: '#ffffff', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Database size={20} style={{ color: '#f59e0b' }} />
          Dataset & Corpus Details
        </h2>
        <p style={{ color: '#cbd5e1', fontSize: '0.95rem', lineHeight: 1.7 }}>
          The system was trained and evaluated on the standardized <strong>SemEval-2018 Task 3 Subtask B</strong> benchmark
          corpus for sarcasm and irony detection in English tweets.
        </p>
        <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '0.5rem', color: '#94a3b8', fontSize: '0.9rem' }}>
          <li style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <CheckCircle2 size={16} style={{ color: '#10b981' }} />
            <span><strong>Task:</strong> 4-class multi-class classification (Regular, Sarcasm, Irony, Figurative).</span>
          </li>
          <li style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <CheckCircle2 size={16} style={{ color: '#10b981' }} />
            <span><strong>Validation Strategy:</strong> Stratified K-Fold cross-validation preserving label distribution.</span>
          </li>
          <li style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <CheckCircle2 size={16} style={{ color: '#10b981' }} />
            <span><strong>Evaluation Metrics:</strong> Macro F1-score (primary), Class-wise Precision/Recall, and Accuracy.</span>
          </li>
        </ul>
      </section>

      {/* 5. Methodology & NLP Pipeline (Section 38 & 43) */}
      <section className="card" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <h2 style={{ fontSize: '1.35rem', color: '#ffffff', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <GitBranch size={20} style={{ color: '#a855f7' }} />
          NLP Pipeline & Methodology
        </h2>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          {[
            { step: '1. Preprocessing', detail: 'URL removal, username neutralization, contraction expansion, lowercase normalization, and punctuation token preservation.' },
            { step: '2. Representation', detail: 'Bag-of-Words (BoW) feature extraction capturing unigram and bigram frequency distributions.' },
            { step: '3. Classification', detail: 'Linear Support Vector Classifier (LinearSVC) utilizing One-vs-Rest decision boundaries.' },
            { step: '4. Probability Calibration', detail: 'Platt scaling / Softmax transformation converting decision function margins into normalized probabilities across all four classes.' },
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
