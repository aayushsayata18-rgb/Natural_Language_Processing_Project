import React, { useState, useEffect, useCallback } from 'react';
import { getPredictionHistory, deletePrediction } from '../services/api';
import PredictionHistory from '../components/PredictionHistory';
import { History as HistoryIcon, RefreshCw, Database } from 'lucide-react';

export function History() {
  const [predictions, setPredictions] = useState([]);
  const [pagination, setPagination] = useState({
    total: 0,
    page: 1,
    limit: 20,
    pages: 1,
  });
  const [selectedClass, setSelectedClass] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchHistory = useCallback(
    async (page = 1, cls = selectedClass) => {
      setLoading(true);
      setError(null);
      try {
        const res = await getPredictionHistory({
          limit: 20,
          page,
          predictedClass: cls,
        });
        setPredictions(res.predictions);
        setPagination(res.pagination);
        setLoading(false);
      } catch (err) {
        setError(err);
        setLoading(false);
      }
    },
    [selectedClass]
  );

  useEffect(() => {
    fetchHistory(1, selectedClass);
  }, [fetchHistory, selectedClass]);

  const handlePageChange = (newPage) => {
    fetchHistory(newPage, selectedClass);
  };

  const handleFilterChange = (newCls) => {
    setSelectedClass(newCls);
    fetchHistory(1, newCls);
  };

  const handleDelete = async (id) => {
    await deletePrediction(id);
    // Refresh current page
    fetchHistory(pagination.page, selectedClass);
  };

  return (
    <div
      className="history-page"
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '1.75rem',
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '1rem',
        }}
      >
        <div>
          <h1
            style={{
              fontSize: '2rem',
              color: '#ffffff',
              display: 'flex',
              alignItems: 'center',
              gap: '0.65rem',
              fontFamily: 'var(--font-heading)',
            }}
          >
            <HistoryIcon size={26} style={{ color: '#6366f1' }} />
            Prediction History
          </h1>
          <p style={{ color: '#94a3b8', fontSize: '0.95rem', marginTop: '0.25rem' }}>
            Historical audit logs of text classifications stored in MongoDB.
          </p>
        </div>

        <button
          onClick={() => fetchHistory(pagination.page, selectedClass)}
          disabled={loading}
          className="btn btn-secondary btn-sm"
          aria-label="Refresh prediction history"
        >
          <RefreshCw size={15} className={loading ? 'animate-spin' : ''} />
          Refresh
        </button>
      </div>

      {/* Main History Table / Component */}
      <PredictionHistory
        predictions={predictions}
        pagination={pagination}
        loading={loading}
        error={error}
        selectedClass={selectedClass}
        onPageChange={handlePageChange}
        onFilterChange={handleFilterChange}
        onDelete={handleDelete}
        onRetry={() => fetchHistory(pagination.page, selectedClass)}
      />
    </div>
  );
}

export default History;
