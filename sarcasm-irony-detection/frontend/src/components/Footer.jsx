import React from 'react';
import { Brain, Layers, Cpu, Database } from 'lucide-react';

export function Footer() {
  return (
    <footer className="footer" role="contentinfo">
      <div className="container footer-inner">
        <div>
          <h3 className="footer-title">
            Sarcasm & Irony Detection System using NLP
          </h3>
          <p className="footer-desc" style={{ marginTop: '0.5rem' }}>
            A four-class Natural Language Processing and Machine Learning classification system
            evaluating figurative and ironic intent from textual discourse.
          </p>
        </div>

        <div className="footer-meta">
          <span style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
            <Brain size={14} style={{ color: '#6366f1' }} /> Model: LinearSVC (SIC-v1.0)
          </span>
          <span>•</span>
          <span style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
            <Cpu size={14} style={{ color: '#38bdf8' }} /> FastAPI ML Service
          </span>
          <span>•</span>
          <span style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
            <Layers size={14} style={{ color: '#10b981' }} /> Node.js Express API
          </span>
          <span>•</span>
          <span style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
            <Database size={14} style={{ color: '#f59e0b' }} /> MongoDB Persistence
          </span>
        </div>

        <div style={{ fontSize: '0.75rem', color: '#64748b' }}>
          NLP Project &copy; {new Date().getFullYear()} — Natural Language Processing & Machine Learning
        </div>
      </div>
    </footer>
  );
}

export default Footer;
