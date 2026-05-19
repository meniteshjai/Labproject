import { useState, useEffect } from 'react';
import { Sun, Moon, UploadCloud, LayoutDashboard, History, CheckCircle, AlertTriangle, AlertCircle, LogOut } from 'lucide-react';
import UploadView from './components/UploadView';
import DashboardView from './components/DashboardView';
import HistoryView from './components/HistoryView';
import LoginView from './components/LoginView';
import { ToastProvider } from './components/Toast';
import './index.css';

function App() {
  const [activeTab, setActiveTab] = useState('upload');
  const [theme, setTheme] = useState(localStorage.getItem('theme') || 'dark');
  const [userEmail, setUserEmail] = useState(localStorage.getItem('chairguardUserEmail') || '');

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem('theme', theme);
  }, [theme]);

  useEffect(() => {
    if (userEmail) {
      localStorage.setItem('chairguardUserEmail', userEmail);
    } else {
      localStorage.removeItem('chairguardUserEmail');
    }
  }, [userEmail]);

  const toggleTheme = () => {
    setTheme(prev => prev === 'dark' ? 'light' : 'dark');
  };

  const handleLogin = (email) => {
    setUserEmail(email);
  };

  const handleLogout = () => {
    setUserEmail('');
    setActiveTab('upload');
  };

  if (!userEmail) {
    return (
      <ToastProvider>
        <main className="main login-screen">
          <LoginView onLogin={handleLogin} />
        </main>
      </ToastProvider>
    );
  }

  return (
    <ToastProvider>
      <header className="header">
        <div className="header-inner">
          <div className="logo" onClick={() => setActiveTab('upload')}>
            <div className="logo-icon">
              <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
                <rect x="4" y="20" width="24" height="3" rx="1.5" fill="url(#grad1)"/>
                <rect x="7" y="10" width="18" height="10" rx="2" fill="url(#grad2)"/>
                <rect x="6" y="23" width="3" height="6" rx="1" fill="url(#grad1)"/>
                <rect x="23" y="23" width="3" height="6" rx="1" fill="url(#grad1)"/>
                <circle cx="26" cy="6" r="5" fill="#00e676" opacity="0.9"/>
                <path d="M24 6l1.5 1.5L28 4.5" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                <defs>
                  <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style={{stopColor: '#6c5ce7'}}/>
                    <stop offset="100%" style={{stopColor: '#a29bfe'}}/>
                  </linearGradient>
                  <linearGradient id="grad2" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style={{stopColor: '#a29bfe'}}/>
                    <stop offset="100%" style={{stopColor: '#6c5ce7'}}/>
                  </linearGradient>
                </defs>
              </svg>
            </div>
            <span className="logo-text">LabEye<span className="logo-ai">AI</span></span>
          </div>

          <nav className="nav">
            <button 
              className={`nav-btn ${activeTab === 'upload' ? 'active' : ''}`}
              onClick={() => setActiveTab('upload')}
            >
              <UploadCloud size={18} />
              <span>Upload & Analyze</span>
            </button>
            <button 
              className={`nav-btn ${activeTab === 'dashboard' ? 'active' : ''}`}
              onClick={() => setActiveTab('dashboard')}
            >
              <LayoutDashboard size={18} />
              <span>Dashboard</span>
            </button>
            <button 
              className={`nav-btn ${activeTab === 'history' ? 'active' : ''}`}
              onClick={() => setActiveTab('history')}
            >
              <History size={18} />
              <span>History</span>
            </button>
          </nav>

          <div className="header-actions">
            <div className="user-chip">{userEmail}</div>
            <button className="btn btn-ghost logout-btn" onClick={handleLogout} title="Log out">
              <LogOut size={18} />
              <span>Logout</span>
            </button>
            <button className="theme-toggle" onClick={toggleTheme} title="Toggle theme">
              {theme === 'dark' ? <Sun size={20} /> : <Moon size={20} />}
            </button>
          </div>
        </div>
      </header>

      <main className="main">
        {activeTab === 'upload' && <UploadView />}
        {activeTab === 'dashboard' && <DashboardView />}
        {activeTab === 'history' && <HistoryView />}
      </main>
    </ToastProvider>
  );
}

export default App;
