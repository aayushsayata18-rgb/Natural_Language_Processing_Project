import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import ErrorBoundary from './components/ErrorBoundary';
import Home from './pages/Home';
import Detector from './pages/Detector';
import History from './pages/History';
import About from './pages/About';

export function App() {
  return (
    <ErrorBoundary>
      <Navbar />
      <main className="main-content" role="main">
        <div className="container">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/detector" element={<Detector />} />
            <Route path="/history" element={<History />} />
            <Route path="/about" element={<About />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </div>
      </main>
      <Footer />
    </ErrorBoundary>
  );
}

export default App;
