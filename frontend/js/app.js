/**
 * Main application logic — handles upload, analysis, results display, navigation, and history.
 */
(function () {
  'use strict';

  const API = '';
  let currentAnalysisId = null;
  let currentResult = null;

  // ===== DOM ELEMENTS =====
  const $ = id => document.getElementById(id);
  const dropzone = $('dropzone');
  const fileInput = $('file-input');
  const previewImg = $('preview-image');
  const dropzoneContent = $('dropzone-content');
  const btnAnalyze = $('btn-analyze');
  const btnClear = $('btn-clear');
  const progressContainer = $('progress-container');
  const progressFill = $('progress-fill');
  const progressText = $('progress-text');
  const resultCard = $('result-card');
  const resultImage = $('result-image');

  let selectedFile = null;

  // ===== NAVIGATION =====
  document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
      btn.classList.add('active');
      const tabId = 'tab-' + btn.dataset.tab;
      $(tabId).classList.add('active');
      if (btn.dataset.tab === 'dashboard') Dashboard.loadStats();
      if (btn.dataset.tab === 'history') loadHistory();
    });
  });

  // ===== THEME TOGGLE =====
  $('theme-toggle').addEventListener('click', () => {
    const html = document.documentElement;
    const next = html.dataset.theme === 'dark' ? 'light' : 'dark';
    html.dataset.theme = next;
    localStorage.setItem('theme', next);
  });
  // Restore saved theme
  const saved = localStorage.getItem('theme');
  if (saved) document.documentElement.dataset.theme = saved;

  // ===== DRAG & DROP =====
  ['dragenter', 'dragover'].forEach(e => dropzone.addEventListener(e, ev => { ev.preventDefault(); dropzone.classList.add('drag-over'); }));
  ['dragleave', 'drop'].forEach(e => dropzone.addEventListener(e, ev => { ev.preventDefault(); dropzone.classList.remove('drag-over'); }));

  dropzone.addEventListener('drop', e => {
    const files = e.dataTransfer.files;
    if (files.length) handleFile(files[0]);
  });

  dropzone.addEventListener('click', () => { if (!selectedFile) fileInput.click(); });
  fileInput.addEventListener('change', () => { if (fileInput.files.length) handleFile(fileInput.files[0]); });

  function handleFile(file) {
    const valid = ['image/jpeg', 'image/jpg', 'image/png'];
    if (!valid.includes(file.type)) { showToast('Please upload a JPG or PNG image.', 'error'); return; }
    if (file.size > 50 * 1024 * 1024) { showToast('File too large. Max 50MB.', 'error'); return; }
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = e => {
      previewImg.src = e.target.result;
      previewImg.classList.remove('hidden');
      dropzoneContent.classList.add('hidden');
    };
    reader.readAsDataURL(file);
    btnAnalyze.disabled = false;
    btnClear.disabled = false;
    resultCard.classList.add('hidden');
  }

  // ===== CLEAR =====
  btnClear.addEventListener('click', () => {
    selectedFile = null;
    fileInput.value = '';
    previewImg.src = '';
    previewImg.classList.add('hidden');
    dropzoneContent.classList.remove('hidden');
    btnAnalyze.disabled = true;
    btnClear.disabled = true;
    resultCard.classList.add('hidden');
    progressContainer.classList.add('hidden');
    currentAnalysisId = null;
    currentResult = null;
  });

  // ===== ANALYZE =====
  btnAnalyze.addEventListener('click', async () => {
    if (!selectedFile) return;
    btnAnalyze.disabled = true;
    progressContainer.classList.remove('hidden');
    progressFill.style.width = '0%';
    progressText.textContent = 'Uploading image...';

    try {
      // Step 1: Upload
      const fd = new FormData();
      fd.append('file', selectedFile);
      fd.append('lab_room', $('lab-room').value);
      fd.append('uploaded_by', $('uploaded-by').value || 'Admin');

      animateProgress(0, 40, 1500);
      const uploadRes = await fetch(API + '/api/upload', { method: 'POST', body: fd });
      const uploadJson = await uploadRes.json();
      if (!uploadJson.success) throw new Error(uploadJson.detail || 'Upload failed');

      currentAnalysisId = uploadJson.data.id;
      progressFill.style.width = '40%';
      progressText.textContent = 'Running AI analysis...';

      // Step 2: Analyze
      animateProgress(40, 90, 5000);
      const analyzeRes = await fetch(API + '/api/analyze/' + currentAnalysisId, { method: 'POST' });
      const analyzeJson = await analyzeRes.json();
      if (!analyzeJson.success) throw new Error(analyzeJson.detail || 'Analysis failed');

      progressFill.style.width = '100%';
      progressText.textContent = 'Done!';
      currentResult = analyzeJson.data;

      setTimeout(() => {
        progressContainer.classList.add('hidden');
        displayResults(currentResult);
        showToast('Analysis completed successfully!', 'success');
      }, 500);

    } catch (err) {
      progressContainer.classList.add('hidden');
      showToast('Error: ' + err.message, 'error');
      btnAnalyze.disabled = false;
    }
  });

  function animateProgress(from, to, duration) {
    const start = Date.now();
    const tick = () => {
      const elapsed = Date.now() - start;
      const pct = Math.min(from + (to - from) * (elapsed / duration), to);
      progressFill.style.width = pct + '%';
      if (elapsed < duration) requestAnimationFrame(tick);
    };
    tick();
  }

  // ===== DISPLAY RESULTS =====
  function displayResults(data) {
    resultCard.classList.remove('hidden');

    // Status badge
    const badge = $('result-status-badge');
    if (data.misplaced_chairs === 0 && data.total_chairs > 0) {
      badge.textContent = 'All Clear'; badge.className = 'status-badge success';
    } else if (data.misplaced_chairs > 0) {
      badge.textContent = 'Issues Found'; badge.className = 'status-badge danger';
    } else {
      badge.textContent = 'No Chairs'; badge.className = 'status-badge warning';
    }

    // Stats
    $('stat-total-chairs').querySelector('.stat-value').textContent = data.total_chairs;
    $('stat-correct').querySelector('.stat-value').textContent = data.correct_chairs;
    $('stat-misplaced').querySelector('.stat-value').textContent = data.misplaced_chairs;
    $('stat-accuracy').querySelector('.stat-value').textContent = data.accuracy + '%';

    // Confidence
    $('confidence-value').textContent = data.avg_confidence + '%';
    setTimeout(() => { $('confidence-fill').style.width = data.avg_confidence + '%'; }, 100);

    // Accuracy ring
    const pct = data.accuracy / 100;
    const circumference = 2 * Math.PI * 52;
    const offset = circumference * (1 - pct);
    const ringFill = $('accuracy-ring-fill');
    ringFill.style.strokeDasharray = circumference;
    setTimeout(() => { ringFill.style.strokeDashoffset = offset; }, 100);
    const ringColor = data.accuracy >= 80 ? '#10b981' : data.accuracy >= 50 ? '#f59e0b' : '#ef4444';
    ringFill.style.stroke = ringColor;
    $('ring-text').textContent = data.accuracy + '%';
    $('ring-text').style.color = ringColor;

    // Images
    if (data.result_image_url) resultImage.src = API + data.result_image_url;
    setupImageTabs(data);

    // Chair list
    renderChairList(data.details?.chairs || []);

    // Download buttons
    $('btn-download-report').onclick = () => { if (data.pdf_report_url) window.open(API + data.pdf_report_url, '_blank'); };
    $('btn-download-image').onclick = () => { if (data.result_image_url) { const a = document.createElement('a'); a.href = API + data.result_image_url; a.download = 'analysis_result.jpg'; a.click(); } };

    // Scroll to results
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function setupImageTabs(data) {
    const tabs = document.querySelectorAll('.image-tab');
    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        tabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        const view = tab.dataset.view;
        if (view === 'annotated' && data.result_image_url) resultImage.src = API + data.result_image_url;
        else if (view === 'heatmap' && data.heatmap_image_url) resultImage.src = API + data.heatmap_image_url;
        else if (view === 'original' && data.upload_image_url) resultImage.src = API + data.upload_image_url;
      });
    });
  }

  function renderChairList(chairs) {
    const list = $('chair-list');
    if (!chairs.length) { list.innerHTML = '<p style="color:var(--text-muted)">No chairs detected</p>'; return; }
    list.innerHTML = chairs.map(c => {
      const ok = c.is_properly_arranged;
      const issues = (c.issues || []).join(', ') || 'Properly positioned';
      return `<div class="chair-item ${ok ? '' : 'misplaced'}">
        <span class="chair-icon">${ok ? '✅' : '❌'}</span>
        <div class="chair-info">
          <div class="chair-name">Chair #${c.chair_id + 1}</div>
          <div class="chair-status">${issues}</div>
        </div>
        <span class="chair-score ${ok ? 'good' : 'bad'}">${Math.round(c.alignment_score)}%</span>
      </div>`;
    }).join('');
  }

  // ===== HISTORY =====
  async function loadHistory() {
    const params = new URLSearchParams();
    const search = $('filter-search')?.value;
    const status = $('filter-status')?.value;
    const lab = $('filter-lab')?.value;
    const date = $('filter-date')?.value;
    if (search) params.set('search', search);
    if (status) params.set('status', status);
    if (lab) params.set('lab_room', lab);
    if (date) params.set('date_from', date);

    try {
      const res = await fetch(API + '/api/history?' + params.toString());
      const json = await res.json();
      const tbody = $('history-body');
      if (!json.data || !json.data.length) {
        tbody.innerHTML = '<tr><td colspan="7" class="empty-state">No analyses found</td></tr>';
        return;
      }
      tbody.innerHTML = json.data.map(a => {
        const date = new Date(a.created_at).toLocaleString();
        const accCls = a.accuracy >= 80 ? 'success' : a.accuracy >= 50 ? 'warning' : 'danger';
        const statusCls = a.status === 'completed' ? 'success' : a.status === 'failed' ? 'danger' : 'pending';
        return `<tr>
          <td>${date}</td>
          <td>${a.upload_image_url ? `<img src="${API + a.upload_image_url}" class="history-thumb" alt="thumb">` : '—'}</td>
          <td>${a.lab_room}</td>
          <td>${a.total_chairs} (${a.misplaced_chairs} ⚠️)</td>
          <td><span class="status-badge ${accCls}">${a.accuracy}%</span></td>
          <td><span class="status-badge ${statusCls}">${a.status}</span></td>
          <td>
            ${a.pdf_report_url ? `<a href="${API + a.pdf_report_url}" target="_blank" class="btn btn-secondary" style="padding:4px 10px;font-size:12px">PDF</a>` : '—'}
          </td>
        </tr>`;
      }).join('');
    } catch (e) {
      console.error('History load error:', e);
    }
  }

  $('btn-filter')?.addEventListener('click', loadHistory);

  // ===== TOAST =====
  function showToast(msg, type = 'info') {
    const container = $('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    const icons = { success: '✅', error: '❌', info: 'ℹ️' };
    toast.innerHTML = `<span>${icons[type] || 'ℹ️'}</span><span>${msg}</span>`;
    container.appendChild(toast);
    setTimeout(() => { toast.style.opacity = '0'; toast.style.transform = 'translateX(100%)'; setTimeout(() => toast.remove(), 300); }, 4000);
  }

  // Initial load
  window.showToast = showToast;
})();
