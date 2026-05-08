/* ── NAVIGATION ──────────────────────────────────────────────────────── */
const navItems = document.querySelectorAll('.nav-item');
const pages    = document.querySelectorAll('.page');

function switchPage(pageId) {
  navItems.forEach(n => n.classList.toggle('active', n.dataset.page === pageId));
  pages.forEach(p => {
    const active = p.id === `page-${pageId}`;
    p.classList.toggle('active', active);
  });
  if (pageId === 'analytics') initCharts();
  if (pageId === 'dataset')   initPieChart();
  closeSidebar();
}

navItems.forEach(item => {
  item.addEventListener('click', () => switchPage(item.dataset.page));
});

/* ── MOBILE SIDEBAR ──────────────────────────────────────────────────── */
const sidebar  = document.getElementById('sidebar');
const overlay  = document.getElementById('sidebarOverlay');
const hamburger = document.getElementById('hamburgerBtn');

function openSidebar()  { sidebar.classList.add('open'); overlay.classList.add('open'); }
function closeSidebar() { sidebar.classList.remove('open'); overlay.classList.remove('open'); }

hamburger.addEventListener('click', openSidebar);
overlay.addEventListener('click', closeSidebar);

/* ── TOAST ───────────────────────────────────────────────────────────── */
const toastEl  = document.getElementById('toast');
const toastMsg = document.getElementById('toastMsg');
const toastIcon = document.getElementById('toastIcon');
let toastTimer;

function showToast(msg, icon = '✅', duration = 3000) {
  toastMsg.textContent = msg;
  toastIcon.textContent = icon;
  toastEl.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.remove('show'), duration);
}

/* ── FILE UPLOAD ─────────────────────────────────────────────────────── */
const uploadZone  = document.getElementById('uploadZone');
const fileInput   = document.getElementById('fileInput');
const previewArea = document.getElementById('previewArea');
const previewImg  = document.getElementById('previewImg');
const analyseBtn  = document.getElementById('analyseBtn');
const clearBtn    = document.getElementById('clearBtn');
let currentFile   = null;

uploadZone.addEventListener('click', () => fileInput.click());
uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f) handleFile(f);
});
fileInput.addEventListener('change', () => { if (fileInput.files[0]) handleFile(fileInput.files[0]); });

function handleFile(file) {
  if (!['image/jpeg','image/jpg','image/png'].includes(file.type)) {
    showToast('Only JPG / PNG files accepted', '⚠️'); return;
  }
  currentFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    uploadZone.style.display = 'none';
    previewArea.style.display = 'block';
    analyseBtn.disabled = false;
    resetResults();
    showToast(`Loaded: ${file.name}`, '📡');
  };
  reader.readAsDataURL(file);
}

clearBtn.addEventListener('click', () => {
  currentFile = null;
  fileInput.value = '';
  previewArea.style.display = 'none';
  uploadZone.style.display = '';
  analyseBtn.disabled = true;
  resetResults();
});

/* ── ANALYSE ─────────────────────────────────────────────────────────── */
analyseBtn.addEventListener('click', async () => {
  if (!currentFile) return;

  analyseBtn.disabled = true;
  analyseBtn.innerHTML = '<span class="spinner"></span> Analysing…';

  const fd = new FormData();
  fd.append('file', currentFile);
  fd.append('engine', document.getElementById('engineSelect').value);

  try {
    const res = await fetch('/api/predict', { method: 'POST', body: fd });
    if (!res.ok) throw new Error(`Server error: ${res.status}`);
    const data = await res.json();
    renderResult(data);
    showToast('Analysis complete', '🔬');
  } catch (err) {
    showToast('Error: ' + err.message, '❌', 5000);
    analyseBtn.disabled = false;
    analyseBtn.innerHTML = 'Run Clinical Analysis';
  }
});

/* ── RENDER RESULT ───────────────────────────────────────────────────── */
function resetResults() {
  document.getElementById('resultPanel').innerHTML = `
    <div class="awaiting">
      <div class="awaiting-icon">🫁</div>
      <div class="awaiting-text">Awaiting radiograph input…</div>
    </div>`;
  document.getElementById('heatmapSection').style.display = 'none';
  analyseBtn.innerHTML = 'Run Clinical Analysis';
}

function renderResult(data) {
  const riskClass  = data.risk === 'CRITICAL' ? 'risk-critical' : data.risk === 'MODERATE' ? 'risk-moderate' : 'risk-low';
  const bannerClass = data.positive ? 'positive' : 'negative';
  const icon  = data.positive ? '🚨' : '✅';

  const reportUrl = `/api/report?score=${data.score}&confidence=${data.confidence}&risk=${data.risk}&report_id=${data.report_id}`;

  document.getElementById('resultPanel').innerHTML = `
    <div class="card" style="animation:fadeUp .5s ease">
      <div class="card-title">
        <svg fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" style="width:14px;height:14px"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
        02 · Neural Verdict
      </div>

      <div class="result-banner ${bannerClass}">
        <div class="result-icon">${icon}</div>
        <div>
          <div class="result-label">${data.diagnosis}</div>
          <div class="result-sub">Report ID: ${data.report_id} · ${data.timestamp}</div>
        </div>
      </div>

      <div class="conf-wrap">
        <div class="conf-header">
          <span class="conf-label">Engine Confidence</span>
          <span class="conf-value">${data.confidence}%</span>
        </div>
        <div class="conf-bar"><div class="conf-fill" id="confFill"></div></div>
      </div>

      <div class="stat-row">
        <div class="stat-pill">
          <div class="stat-pill-val">${data.confidence}%</div>
          <div class="stat-pill-key">Confidence</div>
        </div>
        <div class="stat-pill">
          <div class="stat-pill-val"><span class="risk-badge ${riskClass}">${data.risk}</span></div>
          <div class="stat-pill-key">Risk Level</div>
        </div>
      </div>

      <a class="btn-download" href="${reportUrl}" download>
        <svg fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" style="width:16px;height:16px"><path stroke-linecap="round" stroke-linejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/></svg>
        Download Clinical Report
      </a>

      <p style="font-size:.72rem;color:var(--muted);margin-top:14px;line-height:1.5">
        ⚠️ AI-generated decision support only. Final diagnosis must be confirmed by a certified radiologist.
      </p>
    </div>`;

  // Animate confidence bar
  setTimeout(() => {
    const fill = document.getElementById('confFill');
    if (fill) fill.style.width = data.confidence + '%';
  }, 100);

  analyseBtn.disabled = false;
  analyseBtn.innerHTML = 'Run Clinical Analysis';

  // Show heatmap
  if (data.heatmap_image) {
    document.getElementById('heatmapSection').style.display = 'block';
    document.getElementById('heatmapGrid').innerHTML = `
      <div class="heatmap-img">
        <img src="data:image/jpeg;base64,${data.original_image}" alt="Original X-ray"/>
        <div class="img-label">Original Scan</div>
      </div>
      <div class="heatmap-img">
        <img src="data:image/jpeg;base64,${data.heatmap_image}" alt="Grad-CAM Heatmap"/>
        <div class="img-label">AI Attention Map</div>
      </div>`;
  }
}

/* ── CHARTS ──────────────────────────────────────────────────────────── */
let barChartInst = null;
let pieChartInst = null;

const chartDefaults = {
  color: '#64748b',
  borderColor: 'rgba(255,255,255,0.08)',
  grid: { color: 'rgba(255,255,255,0.05)', drawBorder: false },
  font: { family: 'Inter', size: 12 }
};

function initCharts() {
  if (barChartInst) return;
  const ctx = document.getElementById('barChart');
  if (!ctx) return;

  barChartInst = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
      datasets: [
        {
          label: 'Baseline CNN',
          data: [90, 91, 93, 92],
          backgroundColor: 'rgba(100,116,139,0.5)',
          borderColor: 'rgba(100,116,139,0.8)',
          borderWidth: 1, borderRadius: 6,
        },
        {
          label: 'Pro ResNet50V2',
          data: [93, 92, 98, 95],
          backgroundColor: 'rgba(99,102,241,0.6)',
          borderColor: 'rgba(129,140,248,0.9)',
          borderWidth: 1, borderRadius: 6,
        }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#94a3b8', font: chartDefaults.font } },
        tooltip: { backgroundColor: '#1e293b', titleColor: '#f1f5f9', bodyColor: '#94a3b8' }
      },
      scales: {
        x: { ticks: { color: '#64748b' }, grid: chartDefaults.grid },
        y: { min: 80, max: 100, ticks: { color: '#64748b', callback: v => v + '%' }, grid: chartDefaults.grid }
      }
    }
  });
}

function initPieChart() {
  if (pieChartInst) return;
  const ctx = document.getElementById('pieChart');
  if (!ctx) return;

  pieChartInst = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Normal (25.7%)', 'Pneumonia (74.3%)'],
      datasets: [{
        data: [1341, 3875],
        backgroundColor: ['rgba(34,211,238,0.7)', 'rgba(244,63,94,0.7)'],
        borderColor: ['rgba(34,211,238,1)', 'rgba(244,63,94,1)'],
        borderWidth: 2,
        hoverOffset: 8
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      cutout: '65%',
      plugins: {
        legend: { position: 'bottom', labels: { color: '#94a3b8', padding: 20, font: chartDefaults.font } },
        tooltip: { backgroundColor: '#1e293b', titleColor: '#f1f5f9', bodyColor: '#94a3b8' }
      }
    }
  });
}
