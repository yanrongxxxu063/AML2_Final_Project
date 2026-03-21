// Set Default Chart.js Config for Dark Theme
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.1)';
Chart.defaults.font.family = "'Inter', sans-serif";

let rawData = [];
let charts = {};

document.addEventListener('DOMContentLoaded', () => {
    // Navigation Logic
    const links = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('.dashboard-section');

    links.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            links.forEach(l => l.classList.remove('active'));
            sections.forEach(s => s.classList.replace('active-section', 'hidden'));
            
            link.classList.add('active');
            const target = document.querySelector(link.getAttribute('href'));
            target.classList.replace('hidden', 'active-section');
        });
    });

    // Load Data
    Papa.parse('../Dataset/integrated_panel_with_regimes.csv', {
        download: true,
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: function(results) {
            rawData = results.data;
            document.getElementById('data-status').textContent = "Dashboard Active";
            document.querySelector('.status-dot').classList.add('loaded');
            
            initDashboard();
        },
        error: function(err) {
            console.error("Error loading CSV:", err);
            document.getElementById('data-status').textContent = "Error Loading Data";
            document.querySelector('.status-dot').style.backgroundColor = 'red';
        }
    });

    // Neighborhood select listener
    document.getElementById('neighborhoodSelect').addEventListener('change', (e) => {
        updateHousingChart(e.target.value);
    });
});

function initDashboard() {
    calculateSummaryMetrics();
    buildMacroChart();
    buildRegimeCharts();
    setupHousingControls();
    updateHousingChart('All');
    buildModelsCharts();
}

function calculateSummaryMetrics() {
    const totalTransactions = rawData.reduce((acc, row) => acc + (row.Transaction_Count || 0), 0);
    const uniqueRegimes = new Set(rawData.map(row => row.Regime)).size;

    document.getElementById('val-transactions').textContent = totalTransactions.toLocaleString();
    document.getElementById('val-regimes').textContent = uniqueRegimes;
}

// Helper: Extract Unique Macro Data (since panel has multiple rows per month)
function getUniqueMonthlyMacroData() {
    const seen = new Set();
    const unique = [];
    rawData.forEach(row => {
        // use timestamp string if needed, assuming row.Month is a string or Date
        const m = row.Month;
        if (!seen.has(m) && m) {
            seen.add(m);
            unique.push(row);
        }
    });
    // Sort chronologically
    unique.sort((a,b) => new Date(a.Month) - new Date(b.Month));
    return unique;
}

function buildMacroChart() {
    const macroData = getUniqueMonthlyMacroData();
    const labels = macroData.map(d => {
        const date = new Date(d.Month);
        return `${date.getFullYear()}-${(date.getMonth()+1).toString().padStart(2, '0')}`;
    });

    const fedfunds = macroData.map(d => d.FEDFUNDS);
    const polarity = macroData.map(d => d.LM_Polarity);

    const ctx = document.getElementById('macroChart').getContext('2d');
    charts.macro = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Federal Funds Rate (%)',
                    data: fedfunds,
                    borderColor: '#f43f5e',
                    backgroundColor: 'rgba(244, 63, 94, 0.1)',
                    yAxisID: 'y',
                    tension: 0.3,
                    fill: true
                },
                {
                    label: 'FOMC Sentiment Polarity',
                    data: polarity,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    yAxisID: 'y1',
                    tension: 0.3,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            scales: {
                y: { type: 'linear', display: true, position: 'left', title: { display:true, text:'Rate (%)'} },
                y1: { type: 'linear', display: true, position: 'right', grid: { drawOnChartArea: false }, title: {display:true, text:'Polarity'} }
            }
        }
    });
}

function buildRegimeCharts() {
    const macroData = getUniqueMonthlyMacroData();
    
    // 1. Scatter Chart (FEDFUNDS over time colored by Regime)
    const datasets = [];
    const regimes = [...new Set(macroData.map(d => d.Regime))].sort();
    
    const colors = ['#3b82f6', '#f43f5e', '#10b981', '#8b5cf6', '#f59e0b', '#06b6d4', '#ec4899', '#84cc16'];

    regimes.forEach((r, i) => {
        const rData = macroData.filter(d => d.Regime === r).map(d => ({
            x: new Date(d.Month).getTime(),
            y: d.FEDFUNDS
        }));
        datasets.push({
            label: `Regime ${r}`,
            data: rData,
            backgroundColor: colors[i % colors.length],
            borderColor: colors[i % colors.length],
            pointRadius: 6,
            pointHoverRadius: 8
        });
    });

    const scatterCtx = document.getElementById('regimeScatterChart').getContext('2d');
    charts.scatter = new Chart(scatterCtx, {
        type: 'scatter',
        data: { datasets: datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                    ticks: {
                        callback: function(value) { return new Date(value).getFullYear(); }
                    }
                },
                y: { title: {display:true, text:'Federal Funds Rate (%)'} }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(ctx) {
                            return `Regime ${ctx.dataset.label.replace('Regime ', '')} - FedFunds: ${ctx.raw.y}%`;
                        }
                    }
                }
            }
        }
    });

    // 2. Radar Chart (Average Characteristics of Regimes)
    // We will standardize locally for the radar chart representation
    const features = ['FEDFUNDS', 'UNRATE', 'PCE_YOY', 'MORTGAGE_SPREAD', 'LM_Polarity', 'LM_Subjectivity'];
    
    // Calculate means and std dev for entire dataset
    const stats = {};
    features.forEach(f => {
        const vals = macroData.map(d => d[f]);
        const mean = vals.reduce((a,b)=>a+b,0)/vals.length;
        const std = Math.sqrt(vals.map(v=>Math.pow(v-mean,2)).reduce((a,b)=>a+b,0)/vals.length);
        stats[f] = {mean, std};
    });

    const radarDatasets = regimes.map((r, i) => {
        const rData = macroData.filter(d => d.Regime === r);
        const standardizedMeans = features.map(f => {
            const meanVal = rData.reduce((a,b)=>a+b[f],0)/rData.length;
            // z-score
            return stats[f].std > 0 ? (meanVal - stats[f].mean) / stats[f].std : 0;
        });

        return {
            label: `Regime ${r}`,
            data: standardizedMeans,
            backgroundColor: colors[i % colors.length] + '40', // 25% opacity
            borderColor: colors[i % colors.length],
            pointBackgroundColor: colors[i % colors.length]
        };
    });

    const radarCtx = document.getElementById('regimeRadarChart').getContext('2d');
    charts.radar = new Chart(radarCtx, {
        type: 'radar',
        data: {
            labels: features,
            datasets: radarDatasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    angleLines: { color: 'rgba(255,255,255,0.1)' },
                    grid: { color: 'rgba(255,255,255,0.1)' },
                    pointLabels: { color: '#94a3b8', font: {size: 11} },
                    ticks: { display: false }
                }
            }
        }
    });
}

function setupHousingControls() {
    // Get top 5 neighborhoods by row count
    const nbhoodCounts = {};
    rawData.forEach(row => {
        if(row['Sub-Nbhood']) {
            nbhoodCounts[row['Sub-Nbhood']] = (nbhoodCounts[row['Sub-Nbhood']] || 0) + 1;
        }
    });
    
    const sortedNbhoods = Object.keys(nbhoodCounts).sort((a,b) => nbhoodCounts[b] - nbhoodCounts[a]).slice(0, 5);
    
    const select = document.getElementById('neighborhoodSelect');
    sortedNbhoods.forEach(nb => {
        const opt = document.createElement('option');
        opt.value = nb;
        opt.textContent = nb;
        select.appendChild(opt);
    });
    
    // Save to window for access
    window.topNeighborhoods = sortedNbhoods;
}

function updateHousingChart(filterArg) {
    // A simplified boxplot estimation using bar error bars (Chart.js doesn't natively support boxplots without plugins)
    // We will render a Bar chart showing the Median of Medians, grouped by Regime and Beds.
    
    let filteredData = rawData;
    if(filterArg !== 'All') {
        filteredData = rawData.filter(d => d['Sub-Nbhood'] === filterArg);
    } else {
        filteredData = rawData.filter(d => window.topNeighborhoods.includes(d['Sub-Nbhood']));
    }

    // Target: We want x-Axis = Beds (0, 1, 2, 3+), Series = Regime
    // Beds group config
    filteredData = filteredData.map(d => {
        let b = d.Beds;
        if(b >= 3) b = "3+";
        return {...d, BedGroup: `${b} Bed`};
    });

    const beds = [...new Set(filteredData.map(d => d.BedGroup))].sort();
    const regimes = [...new Set(filteredData.map(d => d.Regime))].sort();
    const colors = ['#3b82f6', '#f43f5e', '#10b981', '#8b5cf6', '#f59e0b', '#06b6d4', '#ec4899', '#84cc16'];

    const datasets = regimes.map((r, i) => {
        const rData = filteredData.filter(d => d.Regime === r);
        const dataVals = beds.map(b => {
            const rbData = rData.filter(d => d.BedGroup === b).map(d=>d.Median_Sale_Price).filter(v=>v>0);
            if(rbData.length === 0) return 0;
            // Return median of these prices
            rbData.sort((x,y)=>x-y);
            const mid = Math.floor(rbData.length/2);
            return rbData.length % 2 !== 0 ? rbData[mid] : (rbData[mid-1] + rbData[mid]) / 2;
        });

        // Convert to log10 for visual scaling as per Phase 3 script
        const logDataVals = dataVals.map(v => v > 0 ? Math.log10(v) : null);

        return {
            label: `Regime ${r}`,
            data: logDataVals,
            backgroundColor: colors[i % colors.length],
            borderRadius: 4
        };
    });

    if(charts.housing) { charts.housing.destroy(); }
    const housingCtx = document.getElementById('housingBoxplotChart').getContext('2d');
    charts.housing = new Chart(housingCtx, {
        type: 'bar',
        data: {
            labels: beds,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    title: { display: true, text: 'Log10(Median Sale Price)' },
                    min: 5, // ~100k
                    max: 7  // ~10m
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(ctx) {
                            const val = ctx.raw;
                            if(val) {
                                // Inverse log10 to format price
                                const actualPrice = Math.pow(10, val);
                                return `Regime ${ctx.dataset.label.replace('Regime ', '')}: $${Math.round(actualPrice).toLocaleString()}`;
                            }
                            return 'No data';
                        }
                    }
                }
            }
        }
    });

}

function buildModelsCharts() {
    // Hardcoded results from python script output in Phase 4
    // (In a real production app we might load this from a JSON payload)
    const rmseLabels = ['Baseline', 'Sentiment-Aware', 'Regime+Sentiment'];
    const rmseData = [0.4581, 0.4578, 0.4190];
    const r2Data = [0.7788, 0.7796, 0.8256];

    // RMSE Chart
    const rmseCtx = document.getElementById('rmseChart').getContext('2d');
    new Chart(rmseCtx, {
        type: 'bar',
        data: {
            labels: rmseLabels,
            datasets: [{
                label: 'Test RMSE',
                data: rmseData,
                backgroundColor: ['#93c5fd', '#60a5fa', '#2563eb'],
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: { y: { min: 0.35, max: 0.50 } }
        }
    });

    // Top Features
    const featLabels = ['Beds_2.0', 'Beds_1.0', 'FEDFUNDS', 'Sub-Nbhood_TriBeCa', 'MORTGAGE_SPREAD', 'PCE_YOY'];
    const featImp = [0.291, 0.223, 0.082, 0.045, 0.041, 0.038];

    const featCtx = document.getElementById('featureChart').getContext('2d');
    new Chart(featCtx, {
        type: 'bar',
        data: {
            labels: featLabels,
            datasets: [{
                label: 'Importance',
                data: featImp,
                backgroundColor: '#f43f5e',
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false
        }
    });
}
