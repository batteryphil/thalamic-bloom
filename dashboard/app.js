// Chart.js global defaults for aesthetic
Chart.defaults.color = '#94a3b8';
Chart.defaults.font.family = "'Inter', sans-serif";

// Chart instances
let lossChart, tpsChart;

// Data arrays
let steps = [];
let losses = [];
let tpsValues = [];
let seenSteps = new Set();

function initCharts() {
    const commonOptions = {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
            duration: 800,
            easing: 'easeOutQuart'
        },
        plugins: {
            legend: { display: false },
            tooltip: {
                backgroundColor: 'rgba(11, 15, 25, 0.9)',
                titleColor: '#fff',
                bodyColor: '#fff',
                borderColor: 'rgba(255,255,255,0.1)',
                borderWidth: 1,
                padding: 12,
                displayColors: false,
            }
        },
        scales: {
            x: {
                grid: { color: 'rgba(255, 255, 255, 0.05)' },
                ticks: { maxTicksLimit: 8 }
            },
            y: {
                grid: { color: 'rgba(255, 255, 255, 0.05)' },
                beginAtZero: false
            }
        },
        elements: {
            point: { radius: 0, hitRadius: 10, hoverRadius: 6 },
            line: { tension: 0.4 } // Smooth curves
        }
    };

    // Loss Chart
    const ctxLoss = document.getElementById('lossChart').getContext('2d');
    const gradientLoss = ctxLoss.createLinearGradient(0, 0, 0, 400);
    gradientLoss.addColorStop(0, 'rgba(0, 242, 254, 0.5)');
    gradientLoss.addColorStop(1, 'rgba(0, 242, 254, 0.0)');

    lossChart = new Chart(ctxLoss, {
        type: 'line',
        data: {
            labels: steps,
            datasets: [{
                label: 'Loss',
                data: losses,
                borderColor: '#00f2fe',
                backgroundColor: gradientLoss,
                borderWidth: 3,
                fill: true
            }]
        },
        options: commonOptions
    });

    // TPS Chart
    const ctxTPS = document.getElementById('tpsChart').getContext('2d');
    const gradientTPS = ctxTPS.createLinearGradient(0, 0, 0, 400);
    gradientTPS.addColorStop(0, 'rgba(79, 172, 254, 0.5)');
    gradientTPS.addColorStop(1, 'rgba(79, 172, 254, 0.0)');

    tpsChart = new Chart(ctxTPS, {
        type: 'line',
        data: {
            labels: steps,
            datasets: [{
                label: 'Tokens/Sec',
                data: tpsValues,
                borderColor: '#4facfe',
                backgroundColor: gradientTPS,
                borderWidth: 3,
                fill: true
            }]
        },
        options: commonOptions
    });
}

async function fetchMetrics() {
    try {
        const response = await fetch('metrics.jsonl?t=' + new Date().getTime()); // cache buster
        if (!response.ok) return;
        
        const text = await response.text();
        const lines = text.trim().split('\n');
        
        let newData = false;
        
        for (const line of lines) {
            if (!line) continue;
            try {
                const data = JSON.parse(line);
                // Always update cards with latest data (even revisited steps)
                document.getElementById('val-step').innerText = data.step.toLocaleString();
                document.getElementById('val-loss').innerText = data.loss.toFixed(4);
                document.getElementById('val-tps').innerText = data.tps.toFixed(0);
                if (data.lr !== undefined) {
                    document.getElementById('val-lr').innerText = parseFloat(data.lr).toExponential(2);
                }
                if (data.entropy !== undefined) {
                    document.getElementById('val-entropy').innerText = data.entropy.toFixed(4);
                }
                if (data.gate_score !== undefined) {
                    document.getElementById('val-gate').innerText = data.gate_score.toFixed(4);
                }
                if (data.salad) {
                    document.getElementById('val-salad').innerText = '> ' + data.salad;
                }

                // Parse the new payload keys (handling varying JSON key names and falling back to 0.0)
                const collapseVal = data.collapse_metric || data.arm_collapse_metric || 0.0;
                const energyVal = data.energy_metric || data.latent_energy || 0.0;

                // 1. Update Latent Energy
                if (energyVal > 0) {
                    document.getElementById('latent-energy-val').innerText = energyVal.toFixed(2);
                }

                // 2. Update Arm Collapse with Visual Diagnostics
                if (collapseVal > 0) {
                    const collapseEl = document.getElementById('collapse-metric-val');
                    const statusEl = document.getElementById('collapse-status');
                    
                    collapseEl.innerText = collapseVal.toFixed(4);

                    // --- DYNAMICAL SYSTEMS COLOR CODING ---
                    if (collapseVal > 0.85) {
                        collapseEl.style.color = '#ff4444'; // RED: Danger (Mode Collapse / Clones)
                        statusEl.innerText = "Status: Monolithic (Redundant)";
                    } else if (collapseVal > 0.60) {
                        collapseEl.style.color = '#ffbb33'; // YELLOW: Healthy Divergence
                        statusEl.innerText = "Status: Splitting (Diverging)";
                    } else {
                        collapseEl.style.color = '#00C851'; // GREEN: Perfect Orthogonality
                        statusEl.innerText = "Status: Specialized MoE";
                    }
                }

                if (!seenSteps.has(data.step)) {
                    seenSteps.add(data.step);
                    steps.push(data.step);
                    losses.push(data.loss);
                    tpsValues.push(data.tps);
                    newData = true;

                    // Keep arrays from growing infinitely (keep last 100)
                    if (steps.length > 100) {
                        steps.shift();
                        losses.shift();
                        tpsValues.shift();
                    }
                }
            } catch (e) {
                console.error("Parse error on line:", line);
            }
        }

        if (newData) {
            lossChart.update();
            tpsChart.update();
        }

    } catch (err) {
        console.error("Failed to fetch metrics:", err);
    }
}

async function fetchHardware() {
    try {
        const response = await fetch('hardware.json?t=' + new Date().getTime());
        if (!response.ok) return;
        
        const data = await response.json();
        
        const gpuEl = document.getElementById('val-gpu-temp');
        const cpuEl = document.getElementById('val-cpu-temp');
        
        gpuEl.innerText = data.gpu_temp.toFixed(1) + ' °C';
        cpuEl.innerText = data.cpu_temp.toFixed(1) + ' °C';
        
        // Dynamic Warning Colors
        if (data.gpu_temp > 80) {
            gpuEl.style.background = 'linear-gradient(to right, #ef4444, #f97316)';
            gpuEl.style.webkitBackgroundClip = 'text';
            gpuEl.style.webkitTextFillColor = 'transparent';
        } else {
            gpuEl.style.background = '';
            gpuEl.style.webkitBackgroundClip = '';
            gpuEl.style.webkitTextFillColor = '';
        }
        
        if (data.cpu_temp > 80) {
            cpuEl.style.background = 'linear-gradient(to right, #ef4444, #f97316)';
            cpuEl.style.webkitBackgroundClip = 'text';
            cpuEl.style.webkitTextFillColor = 'transparent';
        } else {
            cpuEl.style.background = '';
            cpuEl.style.webkitBackgroundClip = '';
            cpuEl.style.webkitTextFillColor = '';
        }
    } catch (err) {
        console.error("Failed to fetch hardware temps:", err);
    }
}

// Initialize and start polling
initCharts();
setInterval(() => {
    fetchMetrics();
    fetchHardware();
}, 2000); // Poll every 2 seconds
fetchMetrics(); // Initial fetch
fetchHardware();
