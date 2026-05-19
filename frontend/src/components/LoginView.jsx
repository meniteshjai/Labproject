import { useState } from 'react';
import { Mail, ShieldCheck } from 'lucide-react';
import { useToast } from './Toast';

const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export default function LoginView({ onLogin }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const toast = useToast();

  const handleSubmit = (event) => {
    event.preventDefault();
    const normalizedEmail = email.trim().toLowerCase();

    if (!EMAIL_PATTERN.test(normalizedEmail)) {
      setError('Please enter a valid email address.');
      toast('Enter a valid email address.', 'error');
      return;
    }

    if (password.length < 6) {
      setError('Password must be at least 6 characters.');
      toast('Password must be at least 6 characters.', 'error');
      return;
    }

    setError('');
    onLogin(normalizedEmail);
    toast('Welcome back!', 'success');
  };

  return (
    <section className="login-card animate-fade-in">
      <div className="login-panel">
        <div className="login-brand">
          <div className="login-icon">
            <Mail size={28} />
          </div>
          <div>
            <h1>ChairGuard AI</h1>
            <p>Secure access for lab chair monitoring and analytics.</p>
          </div>
        </div>

        <form className="login-form" onSubmit={handleSubmit}>
          <div className="form-group">
            <label>Email address</label>
            <input
              type="email"
              placeholder="you@example.com"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              required
            />
          </div>

          <div className="form-group">
            <label>Password</label>
            <input
              type="password"
              placeholder="At least 6 characters"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              required
              minLength={6}
            />
          </div>

          {error && <p className="form-error">{error}</p>}

          <button type="submit" className="btn btn-primary login-submit">
            <ShieldCheck size={16} /> Sign In
          </button>
        </form>
      </div>

      <div className="login-spotlight">
        <div className="login-hero">
          <span>Secure</span>
          <span>Smart</span>
          <span>Insightful</span>
        </div>
        <p>Log in to manage your lab overview, view history, and delete old records safely.</p>
      </div>
    </section>
  );
}
