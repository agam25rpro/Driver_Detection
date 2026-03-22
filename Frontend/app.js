/* API base URL — empty string = same origin (when served from FastAPI at localhost:8000).
   Falls back to an absolute URL only when opening index.html as a local file. */
const API_BASE = (location.protocol === 'file:') ? 'http://localhost:8000' : '';


/* ═══════════════════════════════════════════════════════════════
   STATE
═══════════════════════════════════════════════════════════════ */
let selectedFile    = null;   // File object (for uploads)
let selectedSample  = null;   // Sample filename (for sample picks)
let activeTab       = 'samples';

/* ═══════════════════════════════════════════════════════════════
   INIT
═══════════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  loadSamples();
  loadClasses();
  setupDragDrop();
});

/* ═══════════════════════════════════════════════════════════════
   TABS
═══════════════════════════════════════════════════════════════ */
function switchTab(tab) {
  activeTab = tab;
  document.getElementById('tab-samples').classList.toggle('active', tab === 'samples');
  document.getElementById('tab-upload').classList.toggle('active',  tab === 'upload');
  document.getElementById('panel-samples').classList.toggle('hidden', tab !== 'samples');
  document.getElementById('panel-upload').classList.toggle('hidden',  tab !== 'upload');
  // Reset selection when switching tabs
  resetSelection();
}

/* ═══════════════════════════════════════════════════════════════
   SAMPLE IMAGES
═══════════════════════════════════════════════════════════════ */
async function loadSamples() {
  const grid = document.getElementById('sampleGrid');
  try {
    const res  = await fetch(`${API_BASE}/sample-images`);
    const data = await res.json();
    const samples = data.samples;

    if (!samples || samples.length === 0) {
      grid.innerHTML = '<div class="sample-loading">No sample images found on the server.</div>';
      return;
    }

    grid.innerHTML = '';
    samples.forEach(s => {
      const item = document.createElement('div');
      item.className = 'sample-item';
      item.dataset.filename = s.filename;

      const img = document.createElement('img');
      img.src = `${API_BASE}${s.url}`;
      img.alt = s.filename;
      img.loading = 'lazy';

      item.appendChild(img);
      item.addEventListener('click', () => selectSample(s.filename, item, img.src));
      grid.appendChild(item);
    });

  } catch (err) {
    grid.innerHTML = `<div class="sample-loading">Cannot reach backend at ${API_BASE}.<br/>Start the server with <code>uvicorn main:app --reload</code></div>`;

  }
}

function selectSample(filename, el, imgSrc) {
  // Deselect previously selected
  document.querySelectorAll('.sample-item').forEach(i => i.classList.remove('selected'));
  el.classList.add('selected');

  selectedSample = filename;
  selectedFile   = null;

  // Show preview
  showPreview(imgSrc);
  enableClassify();
}

/* ═══════════════════════════════════════════════════════════════
   FILE UPLOAD
═══════════════════════════════════════════════════════════════ */
function handleFileUpload(e) {
  const file = e.target.files[0];
  if (!file) return;
  selectFile(file);
}

function selectFile(file) {
  selectedFile   = file;
  selectedSample = null;

  const url = URL.createObjectURL(file);
  showPreview(url);
  enableClassify();
}

function setupDragDrop() {
  const dz = document.getElementById('dropzone');
  dz.addEventListener('dragover', e => { e.preventDefault(); dz.style.borderColor = 'var(--accent-1)'; });
  dz.addEventListener('dragleave', ()  => { dz.style.borderColor = ''; });
  dz.addEventListener('drop', e => {
    e.preventDefault();
    dz.style.borderColor = '';
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) selectFile(file);
  });
}

/* ═══════════════════════════════════════════════════════════════
   PREVIEW
═══════════════════════════════════════════════════════════════ */
function showPreview(src) {
  document.getElementById('previewPlaceholder').classList.add('hidden');
  const img = document.getElementById('previewImg');
  img.src = src;
  img.classList.remove('hidden');
  // Hide old results when new image selected
  document.getElementById('resultsPanel').classList.add('hidden');
}

/* ═══════════════════════════════════════════════════════════════
   CLASSIFY
═══════════════════════════════════════════════════════════════ */
function enableClassify() {
  const btn = document.getElementById('classifyBtn');
  btn.disabled = false;
  document.getElementById('btnText').textContent = 'Run Classification';
}

function resetSelection() {
  selectedFile   = null;
  selectedSample = null;
  document.getElementById('classifyBtn').disabled = true;
  document.getElementById('btnText').textContent  = 'Select an image first';
  document.getElementById('previewPlaceholder').classList.remove('hidden');
  document.getElementById('previewImg').classList.add('hidden');
  document.getElementById('resultsPanel').classList.add('hidden');
}

/* Convert an already-loaded <img> element to a Blob via canvas (no network request) */
function imgElToBlob(imgEl) {
  return new Promise((resolve, reject) => {
    try {
      const canvas = document.createElement('canvas');
      canvas.width  = imgEl.naturalWidth  || 300;
      canvas.height = imgEl.naturalHeight || 300;
      canvas.getContext('2d').drawImage(imgEl, 0, 0);
      canvas.toBlob(blob => {
        if (blob) resolve(blob);
        else reject(new Error('Canvas toBlob returned null'));
      }, 'image/jpeg', 0.95);
    } catch (e) { reject(e); }
  });
}

async function runClassification() {
  if (!selectedFile && !selectedSample) return;

  setLoading(true);

  try {
    let blob;

    if (selectedFile) {
      blob = selectedFile;
    } else {
      // Use the already-displayed preview image — no re-fetch needed (avoids CORS for fetch())
      const previewImg = document.getElementById('previewImg');
      blob = await imgElToBlob(previewImg);
    }

    const formData = new FormData();
    formData.append('file', blob, selectedSample || 'upload.jpg');

    const res  = await fetch(`${API_BASE}/predict`, { method: 'POST', body: formData });
    const data = await res.json();

    if (data.status === 'success') {
      renderResults(data);
    } else {
      alert('Prediction failed: ' + JSON.stringify(data));
    }
  } catch (err) {
    alert(`Error contacting backend: ${err.message}`);
  } finally {
    setLoading(false);
  }
}

function setLoading(on) {
  const btn     = document.getElementById('classifyBtn');
  const text    = document.getElementById('btnText');
  const spinner = document.getElementById('btnSpinner');
  btn.disabled  = on;
  text.textContent = on ? 'Classifying…' : 'Run Classification';
  spinner.classList.toggle('hidden', !on);
}

/* ═══════════════════════════════════════════════════════════════
   RENDER RESULTS
═══════════════════════════════════════════════════════════════ */
const CLASS_ICONS = { c0:'', c1:'', c2:'', c3:'', c4:'', c5:'', c6:'', c7:'', c8:'', c9:'' };
const SEVERITY_COLORS = { safe: 'var(--green)', low: 'var(--blue)', medium: 'var(--yellow)', high: 'var(--red)' };
const SEVERITY_LABELS = { safe: 'Safe', low: 'Low Risk', medium: 'Medium Risk', high: 'High Risk' };


function renderResults(data) {
  const panel = document.getElementById('resultsPanel');
  panel.classList.remove('hidden');

  // Top result
  const classId = data.class_id;
  const icon    = CLASS_ICONS[classId] || '🚗';
  const svColor = SEVERITY_COLORS[data.severity] || 'var(--blue)';
  const svLabel = SEVERITY_LABELS[data.severity]  || data.severity;

  const badge = document.getElementById('resultBadge');
  badge.textContent = '';
  badge.className = 'result-badge result-badge--' + data.severity;
  badge.style.background = svColor + '22';


  document.getElementById('resultLabel').textContent = data.prediction;
  document.getElementById('resultLabel').style.color = svColor;
  document.getElementById('resultConf').textContent  = `Confidence: ${data.confidence.toFixed(1)}%`;
  
  const sevEl = document.getElementById('resultSeverity');
  sevEl.textContent = svLabel;
  sevEl.style.background = svColor + '20';
  sevEl.style.color       = svColor;
  sevEl.style.border      = `1px solid ${svColor}44`;

  // Description — find from all_scores
  const found = data.all_scores.find(s => s.id === classId);
  document.getElementById('resultDesc').textContent = found ? found.description : '';

  // Bar chart
  const barsEl = document.getElementById('classBars');
  barsEl.innerHTML = '';
  data.all_scores.forEach(s => {
    const row = document.createElement('div');
    row.className = 'bar-row';

    const lbl = document.createElement('div');
    lbl.className   = 'bar-label';
    lbl.textContent = s.label;
    lbl.title       = s.label;

    const track = document.createElement('div');
    track.className = 'bar-track';

    const fill = document.createElement('div');
    fill.className = 'bar-fill';
    fill.style.background = s.color || 'var(--blue)';
    track.appendChild(fill);

    const pct = document.createElement('div');
    pct.className   = 'bar-pct';
    pct.textContent = s.confidence.toFixed(1) + '%';

    row.append(lbl, track, pct);
    barsEl.appendChild(row);

    // Animate bar after paint
    requestAnimationFrame(() => {
      setTimeout(() => { fill.style.width = Math.min(s.confidence, 100) + '%'; }, 50);
    });
  });

  // Mock warning
  document.getElementById('mockWarning').classList.toggle('hidden', !data.mocked);

  // Scroll into view
  panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/* ═══════════════════════════════════════════════════════════════
   CLASSES GRID
═══════════════════════════════════════════════════════════════ */
async function loadClasses() {
  try {
    const res  = await fetch(`${API_BASE}/classes`);
    const data = await res.json();
    renderClasses(data.classes);
  } catch (e) {
    // If backend not reachable, use fallback data
    renderClasses(FALLBACK_CLASSES);
  }
}

const FALLBACK_CLASSES = [
  {id:"c0",label:"Safe Driving",            severity:"safe",  severity_score:0,color:"#22c55e",description:"Driver is attentive."},
  {id:"c1",label:"Texting — Right",          severity:"high",  severity_score:5,color:"#ef4444",description:"Texting with right hand."},
  {id:"c2",label:"Phone Call — Right",       severity:"high",  severity_score:4,color:"#f97316",description:"Talking on phone, right hand."},
  {id:"c3",label:"Texting — Left",           severity:"high",  severity_score:5,color:"#ef4444",description:"Texting with left hand."},
  {id:"c4",label:"Phone Call — Left",        severity:"high",  severity_score:4,color:"#f97316",description:"Talking on phone, left hand."},
  {id:"c5",label:"Operating Radio",          severity:"medium",severity_score:3,color:"#eab308",description:"Adjusting console controls."},
  {id:"c6",label:"Drinking",                 severity:"medium",severity_score:2,color:"#eab308",description:"Drinking while driving."},
  {id:"c7",label:"Reaching Behind",          severity:"medium",severity_score:3,color:"#eab308",description:"Reaching away from wheel area."},
  {id:"c8",label:"Hair and Makeup",          severity:"medium",severity_score:2,color:"#eab308",description:"Personal grooming while driving."},
  {id:"c9",label:"Talking to Passenger",    severity:"low",   severity_score:1,color:"#3b82f6",description:"Conversing with a passenger."},
];

function renderClasses(classes) {
  const grid = document.getElementById('classesGrid');
  if (!grid) return;
  grid.innerHTML = '';

  classes.forEach(c => {
    const chip = document.createElement('div');
    chip.className = 'class-chip';
    chip.style.borderColor = c.color + '33';

    const scoreStars = '●'.repeat(c.severity_score) + '○'.repeat(5 - c.severity_score);

    chip.innerHTML = `
      <div class="chip-top">
        <span class="chip-id">${c.id.toUpperCase()}</span>
        <span class="chip-dot" style="background:${c.color}"></span>
      </div>
      <div class="chip-label">${c.label}</div>
      <div class="chip-desc">${c.description}</div>
      <div class="chip-score" style="color:${c.color}">${scoreStars} ${c.severity_score}/5</div>
    `;
    grid.appendChild(chip);
  });
}
