import { useState, useEffect } from 'react';
import { Activity, Target, AlertTriangle, Layers } from 'lucide-react';
import { BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { useToast } from './Toast';

export default function DashboardView() {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [theme, setTheme] = useState(localStorage.getItem('theme') || 'dark');
  const toast = useToast();

  useEffect(() => {
    const handleThemeChange = () => {
      const newTheme = localStorage.getItem('theme') || 'dark';
      setTheme(newTheme);
    };
    window.addEventListener('storage', handleThemeChange);
    return () => window.removeEventListener('storage', handleThemeChange);
  }, []);

  // Theme-aware colors
  const isDark = theme === 'dark';
  const chartColors = {
    grid: isDark ? '#333333' : '#e5e7eb',
    axis: isDark ? '#9ca3af' : '#6b7280',
    tooltip: isDark ? '#1f2937' : '#ffffff',
    tooltipBorder: isDark ? '#444444' : '#e5e7eb',
  };

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const res = await fetch('/api/dashboard/stats');
        const data = await res.json();
        if (data.success) {
          setStats(data.data);
        } else {
          toast('Failed to load dashboard statistics', 'error');
        }
      } catch (err) {
        toast(`Error: ${err.message}`, 'error');
      } finally {
        setLoading(false);
      }
    };
    fetchStats();
  }, [toast]);

  if (loading) {
    return <div className="loading-state" style={{ textAlign: 'center', padding: '40px', color: 'var(--text-muted)' }}>Loading dashboard...</div>;
  }

  if (!stats) return null;

  // Transform data for charts using correct backend keys
  const dailyData = stats.daily_stats && Array.isArray(stats.daily_stats) 
    ? stats.daily_stats.reverse().map(d => ({ date: d.date, count: d.count || 0 }))
    : [];
  
  const statusData = [
    { name: 'Completed', value: stats.total_analyses || 0 },
    { name: 'Flagged', value: stats.flagged_cases || 0 },
  ];
  const COLORS = ['#10b981', '#ef4444', '#f59e0b'];

  // Extract accuracy trend from daily stats
  const accuracyData = stats.daily_stats && Array.isArray(stats.daily_stats)
    ? stats.daily_stats.reverse().map((d, i) => ({ index: i, accuracy: Math.round(d.avg_accuracy || 0) }))
    : [];
  
  const labData = stats.lab_stats && Array.isArray(stats.lab_stats) 
    ? stats.lab_stats.map((room) => ({
      room: room.lab_room,
      total: room.count || 0,
      avg_acc: Math.round(room.avg_accuracy || 0),
      misplaced: room.total_misplaced || 0
    }))
    : [];

  return (
    <div className="dashboard-section animate-fade-in">
      <div className="section-header">
        <h1>Analytics Dashboard</h1>
        <p>Monitor lab chair arrangement discipline across all labs</p>
      </div>

      <div className="dashboard-stats">
        <div className="dash-stat-card">
          <div className="dash-stat-icon icon-blue">
            <Activity size={24} />
          </div>
          <div className="dash-stat-info">
            <span className="dash-stat-value">{stats.total_analyses}</span>
            <span className="dash-stat-label">Total Scans</span>
          </div>
        </div>
        <div className="dash-stat-card">
          <div className="dash-stat-icon icon-green">
            <Target size={24} />
          </div>
          <div className="dash-stat-info">
            <span className="dash-stat-value">{Math.round(stats.avg_accuracy)}%</span>
            <span className="dash-stat-label">Avg Accuracy</span>
          </div>
        </div>
        <div className="dash-stat-card">
          <div className="dash-stat-icon icon-red">
            <AlertTriangle size={24} />
          </div>
          <div className="dash-stat-info">
            <span className="dash-stat-value">{stats.total_misplaced}</span>
            <span className="dash-stat-label">Flagged Cases</span>
          </div>
        </div>
        <div className="dash-stat-card">
          <div className="dash-stat-icon icon-purple">
            <Layers size={24} />
          </div>
          <div className="dash-stat-info">
            <span className="dash-stat-value">{stats.total_chairs_detected}</span>
            <span className="dash-stat-label">Chairs Detected</span>
          </div>
        </div>
      </div>

      <div className="charts-grid">
        <div className="card chart-card">
          <div className="card-header"><h2>📊 Daily Analysis Count</h2></div>
          <div className="chart-container" style={{ height: 300 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={dailyData} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
                <XAxis dataKey="date" stroke={chartColors.axis} />
                <YAxis stroke={chartColors.axis} />
                <Tooltip contentStyle={{ backgroundColor: chartColors.tooltip, borderColor: chartColors.tooltipBorder, color: isDark ? '#fff' : '#000' }} />
                <Bar dataKey="count" fill="#6366f1" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        
        <div className="card chart-card">
          <div className="card-header"><h2>🥧 Arrangement Status</h2></div>
          <div className="chart-container" style={{ height: 300 }}>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={statusData} cx="50%" cy="50%" innerRadius={60} outerRadius={100} fill="#8884d8" paddingAngle={5} dataKey="value">
                  {statusData.map((entry, index) => <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />)}
                </Pie>
                <Tooltip contentStyle={{ backgroundColor: chartColors.tooltip, borderColor: chartColors.tooltipBorder, color: isDark ? '#fff' : '#000' }} />
                <Legend wrapperStyle={{ color: chartColors.axis }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card chart-card chart-wide">
          <div className="card-header"><h2>📈 Accuracy Trend</h2></div>
          <div className="chart-container" style={{ height: 300 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={accuracyData} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
                <XAxis dataKey="index" stroke={chartColors.axis} />
                <YAxis domain={[0, 100]} stroke={chartColors.axis} />
                <Tooltip contentStyle={{ backgroundColor: chartColors.tooltip, borderColor: chartColors.tooltipBorder, color: isDark ? '#fff' : '#000' }} />
                <Line type="monotone" dataKey="accuracy" stroke="#3b82f6" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 8 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-header"><h2>🏢 Lab Room Statistics</h2></div>
        <div className="table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Lab Room</th>
                <th>Total Scans</th>
                <th>Avg Accuracy</th>
                <th>Total Misplaced</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {labData.length > 0 ? labData.map((lab, i) => (
                <tr key={i}>
                  <td><strong>{lab.room}</strong></td>
                  <td>{lab.total}</td>
                  <td>{lab.avg_acc}%</td>
                  <td>{lab.misplaced}</td>
                  <td>
                    <span className={`status-badge ${lab.avg_acc >= 80 ? 'success' : lab.avg_acc >= 50 ? 'warning' : 'danger'}`}>
                      {lab.avg_acc >= 80 ? 'Good' : lab.avg_acc >= 50 ? 'Fair' : 'Poor'}
                    </span>
                  </td>
                </tr>
              )) : (
                <tr><td colSpan="5" className="empty-state">No data available yet</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
