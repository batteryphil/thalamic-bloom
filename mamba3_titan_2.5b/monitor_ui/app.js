// ── Chart helpers ──────────────────────────────────────────────────────────
const ACCENT   = '#66fcf1';
const ACCENT2  = '#45a29e';
const WARN     = '#f5a623';
const DANGER   = '#ff4d6d';
const OK       = '#39d353';
const MUTED    = 'rgba(140,146,172,0.4)';
const MAX_PTS  = 200;

const ARM_NAMES = [
    'Anchor','Logic','Recall','Code',
    'Lang','Math','Chat','Fact',
    'Reason','Syn','Sem','Ctx',
    'Plan','Eval','Gen','Aux'
];

function makeChart(id, label, color, opts={}) {
    const ctx = document.getElementById(id).getContext('2d');
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label,
                data: [],
                borderColor: color,
                backgroundColor: `${color}18`,
                borderWidth: 1.5,
                pointRadius: 0,
                fill: true,
                tension: 0.35,
            }]
        },
        options: {
            animation: false,
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: { legend: { display: false }, tooltip: {
                backgroundColor: 'rgba(9,11,15,0.92)',
                borderColor: color,
                borderWidth: 1,
                titleColor: color,
                bodyColor: '#c5c6c7',
                padding: 8,
            }},
            scales: {
                x: { display: false },
                y: {
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    ticks: { color: MUTED, font: { size: 10 }, maxTicksLimit: 5 },
                    ...(opts.yMin !== undefined ? { min: opts.yMin } : {}),
                    ...(opts.yMax !== undefined ? { max: opts.yMax } : {}),
                }
            }
        }
    });
}

function pushPoint(chart, step, value) {
    chart.data.labels.push(step);
    chart.data.datasets[0].data.push(value);
    if (chart.data.labels.length > MAX_PTS) {
        chart.data.labels.shift();
        chart.data.datasets[0].data.shift();
    }
    chart.update('none');
}

// ── Init charts ────────────────────────────────────────────────────────────
const lossChart    = makeChart('lossChart',    'LM Loss',     ACCENT);
const gnormChart   = makeChart('gnormChart',   'Grad Norm',   WARN);
const tpsChart     = makeChart('tpsChart',     'TPS',         OK);
const lrChart      = makeChart('lrChart',      'LR',          '#a78bfa');
const entropyChart = makeChart('entropyChart', 'Entropy',     ACCENT2, { yMin:0, yMax:2.8 });
const tempChart    = makeChart('tempChart',    'GPU °C',      DANGER,  { yMin:40, yMax:90 });

// ── State ──────────────────────────────────────────────────────────────────
let lastStep = -1;
let lossFloor = Infinity;
let totalTargetSteps = 30000;
let seqLen = 768;

// ── Init divergence bars ───────────────────────────────────────────────────
function simToColor(sim) {
    // sim 1.0 = clone = red, 0.0 = specialist = green
    const r = Math.round(sim * 255);
    const g = Math.round((1 - sim) * 180 + 50);
    return `rgb(${r},${g},40)`;
}

function initDivergBars() {
    const barsDiv   = document.getElementById('arm-diverg-bars');
    const labelsDiv = document.getElementById('arm-diverg-labels');
    if (!barsDiv) return;
    barsDiv.innerHTML   = '';
    labelsDiv.innerHTML = '';
    ARM_NAMES.forEach((name, i) => {
        const wrap = document.createElement('div');
        wrap.className = 'arm-diverg-wrap';
        const bar = document.createElement('div');
        bar.className = 'arm-diverg-bar';
        bar.id = `diverg-bar-${i}`;
        bar.style.height = '100%';
        bar.style.background = simToColor(1.0);
        wrap.appendChild(bar);
        barsDiv.appendChild(wrap);

        const lbl = document.createElement('div');
        lbl.className = 'arm-diverg-label';
        lbl.textContent = name;
        labelsDiv.appendChild(lbl);
    });
}
initDivergBars();

function updateDivergBars(sims) {
    if (!sims || sims.length !== 16) return;
    const PHASE3J_THRESH = 0.70;
    let allDiverged = true;
    sims.forEach((sim, i) => {
        const bar = document.getElementById(`diverg-bar-${i}`);
        if (!bar) return;
        const pct = Math.max(3, sim * 100);
        bar.style.height     = `${pct}%`;
        bar.style.background = simToColor(sim);
        bar.title = `${ARM_NAMES[i]}: ${sim.toFixed(4)}`;
        if (sim >= PHASE3J_THRESH) allDiverged = false;
    });
    // Phase 3j badge
    const badge = document.getElementById('phase3j-badge');
    if (badge) badge.classList.toggle('hidden', !allDiverged);
    // Header alert
    const alert = document.getElementById('phase3j-alert');
    if (alert) alert.classList.toggle('hidden', !allDiverged);
}

function updateArmBars(weights) {
    if (!weights || weights.length !== 16) return;
    const maxW = Math.max(...weights);
    weights.forEach((w, i) => {
        const bar = document.getElementById(`arm-bar-${i}`);
        if (!bar) return;
        const pct = Math.max(3, (w / maxW) * 100);
        bar.style.height = `${pct}%`;
        bar.classList.toggle('dominant', w === maxW);
    });
}

// ── Build routing weight bars ──────────────────────────────────────────────
function initArmBars() {
    const barsDiv   = document.getElementById('arm-bars');
    const labelsDiv = document.getElementById('arm-labels');
    if (!barsDiv) return;
    barsDiv.innerHTML = '';
    labelsDiv.innerHTML = '';
    ARM_NAMES.forEach((name, i) => {
        const wrap = document.createElement('div');
        wrap.className = 'arm-bar-wrap';
        const bar = document.createElement('div');
        bar.className = 'arm-bar';
        bar.id = `arm-bar-${i}`;
        bar.style.height = '6.25%';
        wrap.appendChild(bar);
        barsDiv.appendChild(wrap);
        const lbl = document.createElement('div');
        lbl.className = 'arm-label';
        lbl.textContent = name;
        labelsDiv.appendChild(lbl);
    });
}
initArmBars();

async function fetchTelemetry() {
    try {
        const res = await fetch(`telemetry.json?t=${Date.now()}`);
        if (!res.ok) return;
        const t = await res.json();

        if (t.step === lastStep) return;
        lastStep = t.step;

        const phase = t.phase || '?';
        seqLen = phase === '1' ? 1024 : phase === '2' ? 512 : 768;
        totalTargetSteps = (phase === '1') ? 50000 : 30000;

        // ── Header ──────────────────────────────────────────────────────
        document.getElementById('phase-label').textContent = `Phase ${phase} · LIVE`;
        document.getElementById('status-text').textContent = `Step ${t.step.toLocaleString()}`;

        // ── Progress ────────────────────────────────────────────────────
        const pct    = (t.step / totalTargetSteps * 100).toFixed(1);
        const remain = totalTargetSteps - t.step;
        const tps    = parseFloat(t.tps) || 1;
        const etaSec = (remain * seqLen) / tps;
        const eta    = new Date(Date.now() + etaSec * 1000);
        const etaStr = eta.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});
        document.getElementById('progress-label').textContent = `Step ${t.step.toLocaleString()} / ${totalTargetSteps.toLocaleString()}`;
        document.getElementById('progress-pct').textContent   = `${pct}%`;
        document.getElementById('eta-label').textContent      = `ETA ${etaStr}`;
        document.getElementById('progress-fill').style.width  = `${Math.min(pct,100)}%`;
        const tokensSeen = (t.step * seqLen / 1e6).toFixed(1);
        document.getElementById('tokens-processed').textContent = `${tokensSeen}M tokens seen`;

        // ── Loss card ───────────────────────────────────────────────────
        const lm = parseFloat(t.lm_loss);
        if (lm < lossFloor) lossFloor = lm;
        document.getElementById('loss-metric').textContent = lm.toFixed(3);
        document.getElementById('loss-floor').textContent  = `floor ${lossFloor.toFixed(3)}`;

        // ── Grad Norm ───────────────────────────────────────────────────
        const gn = parseFloat(t.grad_norm) || 0;
        const gnCard = document.querySelector('.metric-card:has(#gnorm-metric)');
        document.getElementById('gnorm-metric').textContent = gn.toFixed(2);
        const gnEl = document.getElementById('gnorm-metric').closest('.metric-card');
        gnEl.classList.remove('gnorm-ok','gnorm-warn','gnorm-high');
        if (gn < 5)       { gnEl.classList.add('gnorm-ok');   document.getElementById('gnorm-status').textContent = 'healthy'; }
        else if (gn < 15) { gnEl.classList.add('gnorm-warn'); document.getElementById('gnorm-status').textContent = 'elevated'; }
        else              { gnEl.classList.add('gnorm-high');  document.getElementById('gnorm-status').textContent = '⚠ high'; }

        // ── LR ──────────────────────────────────────────────────────────
        const lr = parseFloat(t.lr) || 0;
        document.getElementById('lr-metric').textContent  = lr.toExponential(1);
        const warmupDone = t.resume_step > 300 || t.step > 500;
        document.getElementById('lr-phase').textContent   = warmupDone ? 'cosine decay' : 'warmup ↑';

        // ── TPS ─────────────────────────────────────────────────────────
        document.getElementById('tps-metric').textContent = Math.round(tps).toLocaleString();
        document.getElementById('tps-seq').textContent    = `seq=${seqLen}`;

        // ── GPU Temp ────────────────────────────────────────────────────
        const temp   = parseInt(t.gpu_temp) || 0;
        const tempEl = document.getElementById('temp-metric').closest('.metric-card');
        document.getElementById('temp-metric').textContent = `${temp}°C`;
        tempEl.classList.remove('temp-ok','temp-warm','temp-hot');
        const tStatus = document.getElementById('temp-status');
        if (temp < 75)      { tempEl.classList.add('temp-ok');   tStatus.textContent = 'cool'; }
        else if (temp < 83) { tempEl.classList.add('temp-warm'); tStatus.textContent = 'warm'; }
        else                { tempEl.classList.add('temp-hot');  tStatus.textContent = '⚠ hot'; }

        // ── Domain Loss ─────────────────────────────────────────────────
        document.getElementById('domain-metric').textContent = parseFloat(t.domain_loss || 0).toFixed(4);

        // ── Entropy ─────────────────────────────────────────────────────
        const ent    = parseFloat(t.entropy) || 0;
        const entPct = (ent / 2.7726 * 100).toFixed(1);
        document.getElementById('entropy-metric').textContent = ent.toFixed(3);
        document.getElementById('entropy-pct').textContent    = `${entPct}% uniform`;

        // ── Gate ────────────────────────────────────────────────────────
        document.getElementById('gate-metric').textContent = (parseFloat(t.gate_score)||0).toFixed(4);

        // ── Arm Collapse (summary numbers) ─────────────────────────────────
        const colMean = parseFloat(t.arm_collapse_mean ?? t.arm_collapse_metric ?? 1.0);
        const colMax  = parseFloat(t.arm_collapse_max  ?? 1.0);
        const meanEl = document.getElementById('collapse-mean');
        const maxEl  = document.getElementById('collapse-max');
        if (meanEl) meanEl.textContent = colMean.toFixed(4);
        if (maxEl)  maxEl.textContent  = `${colMax.toFixed(4)} ${colMax < 0.70 ? '✅' : colMax < 0.85 ? '↓' : '●'}`;

        // ── Per-arm divergence bars ─────────────────────────────────────────
        if (t.arm_sims && t.arm_sims.length === 16) {
            updateDivergBars(t.arm_sims);
        }

        // ── Legacy single collapse card (kept in row 2) ─────────────────────
        const collapseMetric = document.getElementById('collapse-metric');
        if (collapseMetric) collapseMetric.textContent = colMean.toFixed(4);
        const colBar = document.getElementById('collapse-bar');
        if (colBar) colBar.style.width = `${(colMean * 100).toFixed(1)}%`;
        const colCard = document.getElementById('collapse-card');
        if (colCard) {
            if (colMax < 0.70) {
                colCard.classList.add('collapse-ready');
                const cs = document.getElementById('collapse-status');
                if (cs) cs.textContent = '⚡ Phase 3j — ALL arms diverged!';
            } else {
                colCard.classList.remove('collapse-ready');
                const cs = document.getElementById('collapse-status');
                if (cs) cs.textContent = `max ${colMax.toFixed(4)} → last clone diverging`;
            }
        }

        // ── Latent Energy ───────────────────────────────────────────────
        const energy = parseFloat(t.latent_energy) || 0;
        document.getElementById('energy-metric').textContent =
            energy > 1000 ? `${(energy/1000).toFixed(1)}k` : energy.toFixed(1);

        // ── Arm weights ──────────────────────────────────────────────────
        if (t.arm_weights) updateArmBars(t.arm_weights);

        // ── Charts ───────────────────────────────────────────────────────
        const s = t.step;
        pushPoint(lossChart,    s, lm);
        pushPoint(gnormChart,   s, gn);
        pushPoint(tpsChart,     s, tps);
        pushPoint(lrChart,      s, lr);
        pushPoint(entropyChart, s, ent);
        if (t.gpu_temp) pushPoint(tempChart, s, temp);

    } catch(e) { /* silently ignore fetch errors */ }
}

fetchTelemetry();
setInterval(fetchTelemetry, 1000);

// ── Word Salad ─────────────────────────────────────────────────────────────
let lastSaladStep = -1;

async function fetchWordSalad() {
    try {
        const res = await fetch(`word_salad.json?t=${Date.now()}`);
        if (!res.ok) return;
        const data = await res.json();
        if (!data.samples || data.step === lastSaladStep) return;
        lastSaladStep = data.step;

        const qualityEmoji = data.quality === 'good' ? '✅' : data.quality === 'fair' ? '🟡' : '🔴';
        const avgRep = data.avg_rep != null ? ` · rep ${(data.avg_rep*100).toFixed(0)}%` : '';
        document.getElementById('salad-meta').innerText =
            `${qualityEmoji} Step ${data.step.toLocaleString()} · ${data.timestamp} · ${data.elapsed_s}s${avgRep}`;

        const grid = document.getElementById('salad-grid');
        grid.innerHTML = '';
        data.samples.forEach(s => {
            const repRate  = s.rep_rate ?? s.repetition_rate ?? 1;
            const repBadge = repRate < 0.30 ? '✅' : repRate < 0.60 ? '🟡' : '🔴';
            const card = document.createElement('div');
            card.className = 'salad-card fresh';
            card.innerHTML = `
                <div class="salad-prompt">${escapeHtml(s.prompt)}</div>
                <div class="salad-rep">${repBadge} rep ${(repRate*100).toFixed(0)}%</div>
                <div class="salad-output">${escapeHtml(s.output || '(empty)')}</div>
            `;
            grid.appendChild(card);
            card.addEventListener('animationend', () => card.classList.remove('fresh'));
        });
    } catch(e) { /* word_salad.json not yet created */ }
}

function escapeHtml(str) {
    return String(str)
        .replace(/&/g,'&amp;').replace(/</g,'&lt;')
        .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

fetchWordSalad();
setInterval(fetchWordSalad, 5000);
