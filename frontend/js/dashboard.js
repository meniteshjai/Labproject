/**
 * Dashboard module — loads stats and renders dashboard charts/tables.
 */
const Dashboard = {
  async loadStats() {
    try {
      const res = await fetch('/api/dashboard/stats');
      const json = await res.json();
      if (!json.success) return;
      const d = json.data;

      document.getElementById('dash-total-scans').textContent = d.total_analyses || 0;
      document.getElementById('dash-avg-accuracy').textContent = (d.avg_accuracy || 0) + '%';
      document.getElementById('dash-flagged').textContent = d.flagged_cases || 0;
      document.getElementById('dash-total-chairs').textContent = d.total_chairs_detected || 0;

      // Daily bar chart
      const daily = (d.daily_stats || []).reverse();
      if (daily.length) {
        Charts.drawBarChart('chart-daily', daily.map(r => r.date), daily.map(r => r.count), Charts.colors.primary);
      }

      // Status pie chart
      const ok = (d.total_chairs_detected || 0) - (d.total_misplaced || 0);
      const bad = d.total_misplaced || 0;
      Charts.drawPieChart('chart-status', ['Correct', 'Misplaced'], [ok, bad], [Charts.colors.success, Charts.colors.danger]);

      // Accuracy trend line
      if (daily.length) {
        Charts.drawLineChart('chart-trend', daily.map(r => r.date), daily.map(r => Math.round(r.avg_accuracy || 0)), Charts.colors.primary);
      }

      // Lab stats table
      const tbody = document.getElementById('lab-stats-body');
      if (d.lab_stats && d.lab_stats.length) {
        tbody.innerHTML = d.lab_stats.map(lab => {
          const acc = Math.round(lab.avg_accuracy || 0);
          const cls = acc >= 80 ? 'success' : acc >= 50 ? 'warning' : 'danger';
          return `<tr>
            <td><strong>${lab.lab_room}</strong></td>
            <td>${lab.count}</td>
            <td>${acc}%</td>
            <td>${lab.total_misplaced || 0}</td>
            <td><span class="status-badge ${cls}">${cls === 'success' ? 'Good' : cls === 'warning' ? 'Fair' : 'Poor'}</span></td>
          </tr>`;
        }).join('');
      }
    } catch (e) {
      console.error('Dashboard load error:', e);
    }
  }
};
