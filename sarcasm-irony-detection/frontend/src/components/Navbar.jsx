import React, { useState, useEffect } from 'react';
import { NavLink } from 'react-router-dom';
import { checkHealth } from '../services/api';
import { BrainCircuit, Menu, X, Activity } from 'lucide-react';

export function Navbar() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [healthStatus, setHealthStatus] = useState('checking');

  useEffect(() => {
    let isMounted = true;
    const fetchHealth = async () => {
      try {
        const res = await checkHealth();
        if (isMounted) {
          if (res.status === 'healthy') {
            setHealthStatus('healthy');
          } else if (res.status === 'degraded') {
            setHealthStatus('degraded');
          } else {
            setHealthStatus('offline');
          }
        }
      } catch {
        if (isMounted) setHealthStatus('offline');
      }
    };

    fetchHealth();
    // Poll health check once every 30 seconds
    const interval = setInterval(fetchHealth, 30000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  const closeMenu = () => setMobileMenuOpen(false);

  return (
    <header className="navbar" role="banner">
      <div className="container navbar-inner">
        {/* Brand */}
        <NavLink to="/" className="nav-brand" onClick={closeMenu}>
          <div className="nav-brand-icon">
            <BrainCircuit size={19} />
          </div>
          <span>Sarcasm & Irony <span style={{ color: '#6366f1' }}>NLP</span></span>
        </NavLink>

        {/* Navigation Links */}
        <nav
          className={`nav-links ${mobileMenuOpen ? 'open' : ''}`}
          role="navigation"
          aria-label="Main Navigation"
        >
          <NavLink
            to="/"
            end
            className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
            onClick={closeMenu}
          >
            Home
          </NavLink>
          <NavLink
            to="/detector"
            className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
            onClick={closeMenu}
          >
            Detector
          </NavLink>
          <NavLink
            to="/history"
            className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
            onClick={closeMenu}
          >
            History
          </NavLink>
          <NavLink
            to="/about"
            className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
            onClick={closeMenu}
          >
            About
          </NavLink>

          {/* System Health Status Indicator */}
          <div
            className="nav-health"
            title={`Backend Service Status: ${healthStatus}`}
            aria-label={`System status: ${healthStatus}`}
          >
            <span className={`status-dot ${healthStatus}`} />
            <span style={{ textTransform: 'capitalize' }}>
              {healthStatus === 'checking' ? 'Connecting...' : healthStatus}
            </span>
          </div>
        </nav>

        {/* Mobile toggle */}
        <button
          className="mobile-menu-btn"
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          aria-label={mobileMenuOpen ? 'Close navigation menu' : 'Open navigation menu'}
          aria-expanded={mobileMenuOpen}
        >
          {mobileMenuOpen ? <X size={22} /> : <Menu size={22} />}
        </button>
      </div>
    </header>
  );
}

export default Navbar;
