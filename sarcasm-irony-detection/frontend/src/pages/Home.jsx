import React from 'react';
import { Link } from 'react-router-dom';
import {
  Sparkles,
  ArrowRight,
  Brain,
  Layers,
  Cpu,
  Database,
  CheckCircle2,
  BookOpen,
  Sliders,
  BarChart,
  ShieldCheck,
} from 'lucide-react';
import { CLASS_COLORS } from '../utils/formatting';

export function Home() {
  const classesList = ['Regular', 'Sarcasm', 'Irony', 'Figurative'];

  return (
    <div className="home-page" style={{ display: 'flex', flexDirection: 'column', gap: '4rem' }}>
      {/* 1. HERO SECTION (Section 8) */}
      <section
        className="hero-section"
        style={{
          textAlign: 'center',
          padding: '3rem 0 2rem',
          maxWidth: '850px',
          margin: '0 auto',
        }}
      >
        <div
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '0.5rem',
            padding: '0.35rem 0.9rem',
            borderRadius: 'var(--radius-full)',
            background: 'rgba(99, 102, 241, 0.1)',
            border: '1px solid rgba(99, 102, 241, 0.25)',
            color: '#818cf8',
            fontSize: '0.85rem',
            fontWeight: 600,
            marginBottom: '1.5rem',
          }}
        >
          <Sparkles size={15} />
          <span>MCA Final-Year NLP Project — 5-Paradigm Comparative Benchmark</span>
        </div>

        <h1
          style={{
            fontSize: '2.85rem',
            fontWeight: 800,
            marginBottom: '1.25rem',
            letterSpacing: '-0.03em',
            lineHeight: 1.15,
            color: '#ffffff',
          }}
        >
          Sarcasm & Irony Detection <br />
          <span
            style={{
              background: 'linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            Powered by Natural Language Processing
          </span>
        </h1>

        <p
          style={{
            fontSize: '1.15rem',
            color: '#94a3b8',
            lineHeight: 1.6,
            marginBottom: '2.25rem',
          }}
        >
          Analyze sentences and social discourse to discern non-literal intent, sarcastic contempt,
          situational irony, and metaphorical figurative language across <strong>5 benchmarked NLP paradigms</strong> —
          from classical LinearSVC to fine-tuned BERT Transformers.
        </p>

        {/* CTA Buttons */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '1rem',
            flexWrap: 'wrap',
          }}
        >
          <Link to="/detector" className="btn btn-primary" style={{ padding: '0.85rem 1.85rem', fontSize: '1rem' }}>
            <span>Compare Models in Detector</span>
            <ArrowRight size={18} />
          </Link>
          <Link to="/about" className="btn btn-secondary" style={{ padding: '0.85rem 1.85rem', fontSize: '1rem' }}>
            <BookOpen size={18} />
            <span>5-Model Benchmark Report</span>
          </Link>
        </div>
      </section>

      {/* 2. THE 5 BENCHMARK MODELS SHOWCASE */}
      <section style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        <div style={{ textAlign: 'center', maxWidth: '700px', margin: '0 auto' }}>
          <h2 style={{ fontSize: '1.85rem', marginBottom: '0.5rem', color: '#f8fafc' }}>
            5 Comparative NLP Model Architectures
          </h2>
          <p style={{ color: '#94a3b8', fontSize: '0.95rem' }}>
            Our research evaluated five generations of Natural Language Processing methodologies on
            an identical, decontaminated multi-class benchmark. Select any architecture live in the detector!
          </p>
        </div>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
            gap: '1.25rem',
          }}
        >
          {[
            {
              phase: 'Phase 4',
              name: 'LinearSVM (BoW)',
              tag: '⭐ Champion Model',
              tagColor: '#10b981',
              paradigm: 'Bag of Words (n-grams 1-2)',
              params: '112,312',
              macroF1: '0.7187',
              accuracy: '84.57%',
              latency: '0.027 ms',
              desc: 'Highest Macro F1 across all models. Maintains active hyperplane separation on the minority Figurative class without collapsing to zero recall.',
            },
            {
              phase: 'Phase 5',
              name: 'LinearSVM (TF-IDF)',
              tag: 'Statistical NLP',
              tagColor: '#38bdf8',
              paradigm: 'Sublinear TF-IDF + N-grams',
              params: '112,312',
              macroF1: '0.7002',
              accuracy: '85.65%',
              latency: '0.014 ms',
              desc: 'Leverages inverse document frequency to downweight common stopwords while upweighting discriminative sarcastic lexical markers.',
            },
            {
              phase: 'Phase 6',
              name: 'GloVe + Bi-LSTM',
              tag: 'Deep Learning',
              tagColor: '#a855f7',
              paradigm: '100d Dense Word Embeddings',
              params: '3,197,636',
              macroF1: '0.6940',
              accuracy: '86.12%',
              latency: '0.041 ms',
              desc: 'Bidirectional LSTM sequence model learning forward and backward temporal word dependencies from pretrained GloVe vectors.',
            },
            {
              phase: 'Phase 7',
              name: 'FastText + Attention',
              tag: 'Subwords & Attention',
              tagColor: '#f59e0b',
              paradigm: '300d Subwords + Additive Attn',
              params: '9,204,828',
              macroF1: '0.6928',
              accuracy: '86.03%',
              latency: '0.098 ms',
              desc: 'Solves social media out-of-vocabulary slang via character n-grams and attends directly to high-salience contrastive tokens.',
            },
            {
              phase: 'Phase 8',
              name: 'BERT Contextual',
              tag: 'Transformer',
              tagColor: '#ec4899',
              paradigm: '12-Layer Contextual Transformer',
              params: '109,485,316',
              macroF1: '0.6935',
              accuracy: '86.12%',
              latency: '39.33 ms',
              desc: 'Pretrained deep bidirectional transformer fine-tuned end-to-end. Completely eliminated mutual confusion between Sarcasm and Irony.',
            },
          ].map((item, idx) => (
            <div
              key={idx}
              className="card"
              style={{
                border: item.tagColor ? `1px solid ${item.tagColor}40` : '1px solid var(--border-color)',
                display: 'flex',
                flexDirection: 'column',
                gap: '0.85rem',
                position: 'relative',
                background: 'rgba(255, 255, 255, 0.02)',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span
                  style={{
                    fontSize: '0.75rem',
                    fontFamily: 'var(--font-mono)',
                    color: item.tagColor,
                    background: `${item.tagColor}15`,
                    border: `1px solid ${item.tagColor}35`,
                    padding: '0.2rem 0.6rem',
                    borderRadius: 'var(--radius-full)',
                    fontWeight: 600,
                  }}
                >
                  {item.phase} • {item.tag}
                </span>
                <span style={{ fontSize: '0.75rem', color: '#64748b', fontFamily: 'var(--font-mono)' }}>
                  ~{item.latency}
                </span>
              </div>

              <h3 style={{ fontSize: '1.2rem', color: '#ffffff', margin: 0 }}>{item.name}</h3>
              <div style={{ fontSize: '0.8rem', color: '#818cf8', fontFamily: 'var(--font-mono)' }}>
                {item.paradigm}
              </div>

              <p style={{ color: '#94a3b8', fontSize: '0.85rem', lineHeight: 1.5, flex: 1, margin: 0 }}>
                {item.desc}
              </p>

              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(3, 1fr)',
                  gap: '0.5rem',
                  paddingTop: '0.75rem',
                  borderTop: '1px solid var(--border-color)',
                  textAlign: 'center',
                }}
              >
                <div>
                  <div style={{ fontSize: '0.675rem', color: '#64748b', textTransform: 'uppercase' }}>Params</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: '#cbd5e1', fontWeight: 600 }}>
                    {item.params}
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: '0.675rem', color: '#64748b', textTransform: 'uppercase' }}>Macro F1</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: '#10b981', fontWeight: 700 }}>
                    {item.macroF1}
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: '0.675rem', color: '#64748b', textTransform: 'uppercase' }}>Accuracy</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: '#38bdf8', fontWeight: 600 }}>
                    {item.accuracy}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* 3. THE FOUR CLASSES EXPLAINED (Section 9) */}
      <section style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        <div style={{ textAlign: 'center', maxWidth: '650px', margin: '0 auto' }}>
          <h2 style={{ fontSize: '1.75rem', marginBottom: '0.5rem', color: '#f8fafc' }}>
            Four-Class Discourse Taxonomy
          </h2>
          <p style={{ color: '#94a3b8', fontSize: '0.95rem' }}>
            The classification system categorizes textual expressions into four rigorous, mutually
            exclusive linguistic categories established by benchmark NLP research.
          </p>
        </div>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
            gap: '1.25rem',
          }}
        >
          {classesList.map((cls) => {
            const info = CLASS_COLORS[cls];
            return (
              <div
                key={cls}
                className="card"
                style={{
                  border: `1px solid ${info.hex}30`,
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.85rem',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <span className={`badge ${info.badgeClass}`}>{cls}</span>
                  <span
                    style={{
                      width: '10px',
                      height: '10px',
                      borderRadius: '50%',
                      background: info.hex,
                      boxShadow: `0 0 8px ${info.hex}`,
                    }}
                  />
                </div>
                <h3 style={{ fontSize: '1.15rem', color: '#ffffff' }}>{info.label}</h3>
                <p style={{ color: '#94a3b8', fontSize: '0.875rem', lineHeight: 1.5, flex: 1 }}>
                  {info.description}
                </p>
              </div>
            );
          })}
        </div>
      </section>

      {/* 4. HOW IT WORKS 5-STEP PIPELINE (Section 10) */}
      <section className="card" style={{ padding: '2.5rem 2rem' }}>
        <div style={{ textAlign: 'center', maxWidth: '600px', margin: '0 auto 2.5rem' }}>
          <h2 style={{ fontSize: '1.75rem', marginBottom: '0.5rem', color: '#ffffff' }}>
            How the System Works
          </h2>
          <p style={{ color: '#94a3b8', fontSize: '0.95rem' }}>
            End-to-end NLP processing flow from raw user input to multi-class probability scoring.
          </p>
        </div>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
            gap: '1.25rem',
            position: 'relative',
          }}
        >
          {[
            {
              step: 'STEP 1',
              title: 'Text Input',
              desc: 'User inputs sentence or tweet into the interface.',
              icon: <Sparkles size={22} style={{ color: '#6366f1' }} />,
            },
            {
              step: 'STEP 2',
              title: 'NLP Preprocessing',
              desc: 'Case normalization, contraction expansion, and tokenization.',
              icon: <Sliders size={22} style={{ color: '#38bdf8' }} />,
            },
            {
              step: 'STEP 3',
              title: 'Model Representation',
              desc: 'Extracts BoW tokens, TF-IDF weights, GloVe vectors, FastText subwords, or WordPiece IDs.',
              icon: <Cpu size={22} style={{ color: '#a855f7' }} />,
            },
            {
              step: 'STEP 4',
              title: 'Multi-Model Inference',
              desc: 'Executes chosen model (LinearSVC, Bi-LSTM, Attention, or BERT) with calibrated softmax.',
              icon: <Brain size={22} style={{ color: '#ec4899' }} />,
            },
            {
              step: 'STEP 5',
              title: 'Display & History',
              desc: 'Live UI presentation with confidence metrics and MongoDB persistence.',
              icon: <BarChart size={22} style={{ color: '#10b981' }} />,
            },
          ].map((item, idx) => (
            <div
              key={idx}
              style={{
                background: 'rgba(255, 255, 255, 0.02)',
                border: '1px solid var(--border-color)',
                borderRadius: '12px',
                padding: '1.5rem 1.25rem',
                display: 'flex',
                flexDirection: 'column',
                gap: '0.75rem',
              }}
            >
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                }}
              >
                <div
                  style={{
                    width: '42px',
                    height: '42px',
                    borderRadius: '10px',
                    background: 'rgba(255, 255, 255, 0.05)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  {item.icon}
                </div>
                <span
                  style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.75rem',
                    color: '#64748b',
                    fontWeight: 600,
                  }}
                >
                  {item.step}
                </span>
              </div>
              <h4 style={{ fontSize: '1rem', color: '#f8fafc' }}>{item.title}</h4>
              <p style={{ color: '#94a3b8', fontSize: '0.825rem', lineHeight: 1.5 }}>
                {item.desc}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* 4. ARCHITECTURE OVERVIEW CALLOUT (Section 2 & 89) */}
      <section
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
          gap: '1.5rem',
        }}
      >
        <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <h3 style={{ fontSize: '1.25rem', color: '#ffffff', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Layers size={20} style={{ color: '#6366f1' }} />
            Decoupled Multi-Tier System
          </h3>
          <p style={{ color: '#94a3b8', fontSize: '0.9rem', lineHeight: 1.6 }}>
            Designed strictly under industry standards: React provides the presentation layer,
            communicating solely via REST with Node.js Express. Express orchestrates Python
            FastAPI for ML inference while asynchronously storing audit history into MongoDB.
          </p>
          <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '0.5rem', marginTop: 'auto' }}>
            <li style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#cbd5e1', fontSize: '0.85rem' }}>
              <CheckCircle2 size={16} style={{ color: '#10b981' }} /> Zero direct React-to-Python communication
            </li>
            <li style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#cbd5e1', fontSize: '0.85rem' }}>
              <CheckCircle2 size={16} style={{ color: '#10b981' }} /> Rate limiting & input validation guards
            </li>
            <li style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#cbd5e1', fontSize: '0.85rem' }}>
              <CheckCircle2 size={16} style={{ color: '#10b981' }} /> Checksum-verified offline model packaging
            </li>
          </ul>
        </div>

        <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <h3 style={{ fontSize: '1.25rem', color: '#ffffff', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <ShieldCheck size={20} style={{ color: '#38bdf8' }} />
            Academic Integrity & Ethics
          </h3>
          <p style={{ color: '#94a3b8', fontSize: '0.9rem', lineHeight: 1.6 }}>
            The system treats natural language classification as a probabilistic output based on training
            distributions. It does not fabricate subjective claims about human emotion or intent, adhering
            to NLP ethics standards.
          </p>
          <div
            style={{
              marginTop: 'auto',
              padding: '0.85rem 1rem',
              borderRadius: '8px',
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid var(--border-color)',
              fontSize: '0.825rem',
              color: '#94a3b8',
            }}
          >
            LinearSVC achieved a Macro F1 score of <strong>0.7719</strong> and Accuracy of <strong>77.00%</strong> in comprehensive Phase 9 benchmarks.
          </div>
        </div>
      </section>

      {/* 5. FINAL CTA BANNER */}
      <section
        style={{
          background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(168, 85, 247, 0.15) 100%)',
          border: '1px solid rgba(99, 102, 241, 0.3)',
          borderRadius: 'var(--radius-lg)',
          padding: '2.5rem 2rem',
          textAlign: 'center',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '1.25rem',
        }}
      >
        <h2 style={{ fontSize: '1.85rem', color: '#ffffff' }}>
          Ready to Test Sarcasm & Irony Classification?
        </h2>
        <p style={{ color: '#cbd5e1', maxWidth: '580px', fontSize: '0.95rem' }}>
          Enter any text snippet or pick from curated linguistic examples to examine confidence scores and calibrated class probabilities.
        </p>
        <Link to="/detector" className="btn btn-primary" style={{ padding: '0.85rem 2rem' }}>
          <span>Launch Detector</span>
          <ArrowRight size={18} />
        </Link>
      </section>
    </div>
  );
}

export default Home;
