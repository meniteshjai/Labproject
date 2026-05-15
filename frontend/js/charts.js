/**
 * Charts module — renders analytics charts using Canvas API.
 * No external chart libraries needed.
 */
const Charts = {
  colors: {
    primary: '#6366f1', primaryLight: '#818cf8',
    success: '#10b981', danger: '#ef4444',
    warning: '#f59e0b', info: '#3b82f6',
    purple: '#8b5cf6', grid: 'rgba(148,163,184,0.1)',
    text: '#94a3b8', textLight: '#64748b'
  },

  _getCtx(canvasId) {
    const c = document.getElementById(canvasId);
    if (!c) return null;
    const dpr = window.devicePixelRatio || 1;
    const rect = c.getBoundingClientRect();
    c.width = rect.width * dpr;
    c.height = rect.height * dpr;
    const ctx = c.getContext('2d');
    ctx.scale(dpr, dpr);
    c.style.width = rect.width + 'px';
    c.style.height = rect.height + 'px';
    return { ctx, w: rect.width, h: rect.height };
  },

  drawBarChart(canvasId, labels, values, color = this.colors.primary) {
    const r = this._getCtx(canvasId);
    if (!r) return;
    const { ctx, w, h } = r;
    const pad = { top: 20, right: 20, bottom: 40, left: 50 };
    const cw = w - pad.left - pad.right;
    const ch = h - pad.top - pad.bottom;
    const max = Math.max(...values, 1);

    // Grid
    ctx.strokeStyle = this.colors.grid;
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + ch - (ch / 4) * i;
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke();
      ctx.fillStyle = this.colors.textLight; ctx.font = '11px Inter';
      ctx.textAlign = 'right';
      ctx.fillText(Math.round(max / 4 * i), pad.left - 8, y + 4);
    }

    // Bars
    const barW = Math.min(cw / labels.length * 0.6, 40);
    const gap = cw / labels.length;
    labels.forEach((label, i) => {
      const x = pad.left + gap * i + gap / 2 - barW / 2;
      const barH = (values[i] / max) * ch;
      const y = pad.top + ch - barH;
      const gradient = ctx.createLinearGradient(x, y, x, pad.top + ch);
      gradient.addColorStop(0, color);
      gradient.addColorStop(1, color + '40');
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.roundRect(x, y, barW, barH, [4, 4, 0, 0]);
      ctx.fill();
      // Label
      ctx.fillStyle = this.colors.text; ctx.font = '10px Inter'; ctx.textAlign = 'center';
      ctx.fillText(label.slice(5) || label, pad.left + gap * i + gap / 2, h - 10);
    });
  },

  drawPieChart(canvasId, labels, values, colors) {
    const r = this._getCtx(canvasId);
    if (!r) return;
    const { ctx, w, h } = r;
    const cx = w / 2; const cy = h / 2 - 10;
    const radius = Math.min(cx, cy) - 30;
    const total = values.reduce((a, b) => a + b, 0) || 1;
    let startAngle = -Math.PI / 2;

    values.forEach((val, i) => {
      const sliceAngle = (val / total) * 2 * Math.PI;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.arc(cx, cy, radius, startAngle, startAngle + sliceAngle);
      ctx.closePath();
      ctx.fillStyle = colors[i] || this.colors.primary;
      ctx.fill();
      // Label
      if (val > 0) {
        const midAngle = startAngle + sliceAngle / 2;
        const lx = cx + Math.cos(midAngle) * (radius * 0.65);
        const ly = cy + Math.sin(midAngle) * (radius * 0.65);
        ctx.fillStyle = '#fff'; ctx.font = 'bold 13px Inter'; ctx.textAlign = 'center';
        ctx.fillText(Math.round(val / total * 100) + '%', lx, ly + 5);
      }
      startAngle += sliceAngle;
    });
    // Center hole (donut)
    ctx.beginPath(); ctx.arc(cx, cy, radius * 0.5, 0, Math.PI * 2); ctx.fillStyle = getComputedStyle(document.body).getPropertyValue('--bg-card').trim() || '#111827'; ctx.fill();
    // Legend
    const ly = h - 20;
    let lx = cx - (labels.length * 70) / 2;
    labels.forEach((l, i) => {
      ctx.fillStyle = colors[i]; ctx.fillRect(lx, ly - 6, 10, 10);
      ctx.fillStyle = this.colors.text; ctx.font = '11px Inter'; ctx.textAlign = 'left';
      ctx.fillText(l, lx + 14, ly + 4);
      lx += 80;
    });
  },

  drawLineChart(canvasId, labels, values, color = this.colors.primary) {
    const r = this._getCtx(canvasId);
    if (!r) return;
    const { ctx, w, h } = r;
    const pad = { top: 20, right: 20, bottom: 40, left: 50 };
    const cw = w - pad.left - pad.right;
    const ch = h - pad.top - pad.bottom;
    const max = Math.max(...values, 100);
    const min = Math.min(...values, 0);
    const range = max - min || 1;

    // Grid
    ctx.strokeStyle = this.colors.grid; ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + ch - (ch / 4) * i;
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke();
      ctx.fillStyle = this.colors.textLight; ctx.font = '11px Inter'; ctx.textAlign = 'right';
      ctx.fillText(Math.round(min + range / 4 * i) + '%', pad.left - 8, y + 4);
    }

    if (values.length < 2) return;
    const step = cw / (values.length - 1);

    // Area fill
    const gradient = ctx.createLinearGradient(0, pad.top, 0, pad.top + ch);
    gradient.addColorStop(0, color + '30');
    gradient.addColorStop(1, color + '00');
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top + ch);
    values.forEach((v, i) => {
      const x = pad.left + step * i;
      const y = pad.top + ch - ((v - min) / range) * ch;
      ctx.lineTo(x, y);
    });
    ctx.lineTo(pad.left + step * (values.length - 1), pad.top + ch);
    ctx.closePath();
    ctx.fillStyle = gradient; ctx.fill();

    // Line
    ctx.beginPath();
    ctx.strokeStyle = color; ctx.lineWidth = 2.5; ctx.lineJoin = 'round';
    values.forEach((v, i) => {
      const x = pad.left + step * i;
      const y = pad.top + ch - ((v - min) / range) * ch;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Dots
    values.forEach((v, i) => {
      const x = pad.left + step * i;
      const y = pad.top + ch - ((v - min) / range) * ch;
      ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fillStyle = color; ctx.fill();
      ctx.beginPath(); ctx.arc(x, y, 2, 0, Math.PI * 2);
      ctx.fillStyle = '#fff'; ctx.fill();
    });

    // X labels
    labels.forEach((l, i) => {
      if (labels.length > 15 && i % 3 !== 0) return;
      ctx.fillStyle = this.colors.text; ctx.font = '10px Inter'; ctx.textAlign = 'center';
      ctx.fillText(l.slice(5) || l, pad.left + step * i, h - 10);
    });
  }
};
