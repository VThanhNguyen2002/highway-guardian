<template>
  <div class="charts-grid">
    <!-- Pie: super-category distribution -->
    <div class="chart-card">
      <h4 class="chart-title">
        <vue-feather type="pie-chart" size="15" />
        Phân loại Biển báo (Nhóm)
      </h4>
      <div class="chart-wrap doughnut-wrap">
        <Doughnut v-if="hasData" :data="categoryData" :options="doughnutOpts" />
        <p v-else class="no-data">Chưa có dữ liệu</p>
      </div>
    </div>

    <!-- Bar: top-10 individual sign frequency -->
    <div class="chart-card chart-wide">
      <h4 class="chart-title">
        <vue-feather type="bar-chart" size="15" />
        Top {{ topN }} Biển hay gặp nhất
      </h4>
      <div class="chart-wrap">
        <Bar v-if="hasData" :data="frequencyData" :options="barOpts" />
        <p v-else class="no-data">Chưa có dữ liệu</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue';
import VueFeather from 'vue-feather';
import { Bar, Doughnut } from 'vue-chartjs';
import {
  Chart as ChartJS, CategoryScale, LinearScale, BarElement,
  ArcElement, Tooltip, Legend,
} from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, BarElement, ArcElement, Tooltip, Legend);

const props = defineProps({
  docs: { type: Array, default: () => [] },
});

const topN = 10;

// ── Zalo AI 2020 8-class → Super-Category mapping ──────────────────────────
// Exact label strings from src/core/mapping.py SIGN_NAMES:
//   0 = "Background"                  → (skipped)
//   1 = "Cấm ngược chiều"             → Biển cấm
//   2 = "Cấm dừng và đỗ"              → Biển cấm
//   3 = "Cấm rẽ"                      → Biển cấm
//   4 = "Giới hạn tốc độ"             → Biển cấm  (speed limit is a prohibitory sign)
//   5 = "Cấm ô tô"                    → Biển cấm
//   6 = "Cấm đỗ"                      → Biển cấm
//   7 = "Cấm các phương tiện khác"    → Biển cấm
const CLASS_TO_SUPER = {
  // ── Core 8-class backend labels (exact match) ──────────────────────────
  'Cấm ngược chiều':            'Biển cấm',
  'Cấm dừng và đỗ':             'Biển cấm',
  'Cấm rẽ':                     'Biển cấm',
  'Giới hạn tốc độ':            'Biển cấm',
  'Cấm ô tô':                   'Biển cấm',
  'Cấm đỗ':                     'Biển cấm',
  'Cấm các phương tiện khác':   'Biển cấm',
  // ── Manual edit aliases (from Detection Table edit modal) ──────────────
  'Biển cấm':                   'Biển cấm',
  'Biển hiệu lệnh':             'Biển hiệu lệnh',
  'Biển nguy hiểm':             'Biển nguy hiểm',
  'Biển cảnh báo':              'Biển nguy hiểm',
  'Biển chỉ dẫn':               'Khác',
  'Biển phụ':                   'Khác',
  'Hướng đi':                   'Khác',
  'Không xác định':             'Khác',
};

/**
 * Normalise a raw Firestore label string to a super-category.
 * Guarantees no "Undefined" / null ever reaches the chart labels.
 */
function normalizeLabel(raw) {
  if (!raw || raw === 'Background') return null;   // skip background docs entirely
  const trimmed = raw.trim();
  // Backend dynamic fallback pattern: "Không xác định (ID=X)"
  if (trimmed.startsWith('Không xác định')) return 'Khác';
  // Direct lookup — fallback to "Khác" (never "Undefined")
  return CLASS_TO_SUPER[trimmed] ?? 'Khác';
}

const SUPER_COLORS = {
  'Biển cấm':        { bg: '#ef4444cc', border: '#ef4444' },
  'Biển hiệu lệnh':  { bg: '#3b82f6cc', border: '#3b82f6' },
  'Biển nguy hiểm':  { bg: '#f59e0bcc', border: '#f59e0b' },
  'Khác':            { bg: '#94a3b8cc', border: '#94a3b8' },
};

const PAL = ['#667eea','#764ba2','#22c55e','#f59e0b','#ef4444',
             '#06b6d4','#8b5cf6','#ec4899','#14b8a6','#f97316'];

const hasData = computed(() => props.docs.length > 0);

// Super-category frequency
const categoryFreq = computed(() => {
  const freq = {};
  props.docs.forEach(d => {
    const cat = normalizeLabel(d.label);
    if (!cat) return;                                 // skip background docs
    freq[cat] = (freq[cat] || 0) + 1;
  });
  return Object.entries(freq).sort((a, b) => b[1] - a[1]);
});

const categoryData = computed(() => ({
  labels: categoryFreq.value.map(([k]) => k),
  datasets: [{
    data:            categoryFreq.value.map(([, v]) => v),
    backgroundColor: categoryFreq.value.map(([k]) => SUPER_COLORS[k]?.bg  ?? '#94a3b8cc'),
    borderColor:     categoryFreq.value.map(([k]) => SUPER_COLORS[k]?.border ?? '#94a3b8'),
    borderWidth: 2,
    hoverOffset: 8,
  }],
}));

const doughnutOpts = {
  responsive: true,
  maintainAspectRatio: false,
  cutout: '60%',
  plugins: {
    legend: { position: 'bottom', labels: { font: { size: 11 }, padding: 12, color: '#4a5568' } },
    tooltip: { callbacks: { label: (ctx) => ` ${ctx.label}: ${ctx.parsed}` } },
  },
};

// Top-N individual sign frequency
const signFreq = computed(() => {
  const freq = {};
  props.docs.forEach(d => { const k = d.label || 'Unknown'; freq[k] = (freq[k] || 0) + 1; });
  return Object.entries(freq).sort((a, b) => b[1] - a[1]).slice(0, topN);
});

const frequencyData = computed(() => ({
  labels: signFreq.value.map(([k]) => k.length > 22 ? k.slice(0, 20) + '…' : k),
  datasets: [{
    label: 'Lần phát hiện',
    data: signFreq.value.map(([, v]) => v),
    backgroundColor: signFreq.value.map((_, i) => PAL[i % PAL.length] + 'cc'),
    borderColor:     signFreq.value.map((_, i) => PAL[i % PAL.length]),
    borderWidth: 1,
    borderRadius: 5,
  }],
}));

const barOpts = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: { display: false },
    tooltip: { callbacks: { label: (ctx) => ` ${ctx.parsed.y} lần` } },
  },
  scales: {
    x: { grid: { display: false }, ticks: { font: { size: 10 }, color: '#718096' } },
    y: { beginAtZero: true, ticks: { stepSize: 1, font: { size: 10 }, color: '#718096' },
         grid: { color: 'rgba(0,0,0,0.04)' } },
  },
};
</script>

<style scoped>
.charts-grid {
  display: grid;
  grid-template-columns: 320px 1fr;
  gap: 20px;
}
@media (max-width: 900px) { .charts-grid { grid-template-columns: 1fr; } }

.chart-card {
  background: #fff;
  border: 1px solid #e2e8f0;
  border-radius: 14px;
  padding: 20px 24px;
  box-shadow: 0 1px 3px rgba(0,0,0,.05);
}

.chart-title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.88rem;
  font-weight: 700;
  color: #2d3748;
  margin-bottom: 14px;
  padding-bottom: 10px;
  border-bottom: 1px solid #f0f0f0;
}

.chart-wrap { height: 260px; position: relative; }
.doughnut-wrap { height: 280px; }

.no-data {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #a0aec0;
  font-size: 0.88rem;
  font-style: italic;
}
</style>
