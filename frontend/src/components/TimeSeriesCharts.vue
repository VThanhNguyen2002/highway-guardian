<template>
  <div class="chart-card">
    <div class="chart-header">
      <h4 class="chart-title">
        <vue-feather type="trending-up" size="15" />
        Xu hướng Phát hiện (7 ngày gần nhất)
      </h4>
      <div class="period-toggle">
        <button :class="{ active: period === '7d' }" @click="period = '7d'">7 ngày</button>
        <button :class="{ active: period === '30d' }" @click="period = '30d'">30 ngày</button>
      </div>
    </div>
    <div class="chart-wrap">
      <Line v-if="hasData" :data="lineData" :options="lineOpts" />
      <p v-else class="no-data">Chưa có dữ liệu thời gian thực</p>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue';
import VueFeather from 'vue-feather';
import { Line } from 'vue-chartjs';
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement,
  LineElement, Filler, Tooltip, Legend,
} from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Filler, Tooltip, Legend);

const props = defineProps({
  docs: { type: Array, default: () => [] },
});

const period = ref('7d');
const days   = computed(() => period.value === '7d' ? 7 : 30);

// Build ordered list of the last N date strings
const dateLabels = computed(() => {
  const list = [];
  for (let i = days.value - 1; i >= 0; i--) {
    const d = new Date();
    d.setDate(d.getDate() - i);
    list.push(d.toLocaleDateString('vi-VN', { month: 'short', day: 'numeric' }));
  }
  return list;
});

// Map each doc to its date label (vi-VN locale)
const dateCounts = computed(() => {
  const counts = Object.fromEntries(dateLabels.value.map(l => [l, 0]));
  const cutoff  = new Date();
  cutoff.setDate(cutoff.getDate() - days.value + 1);
  cutoff.setHours(0, 0, 0, 0);

  props.docs.forEach(d => {
    if (!d.timestamp) return;
    const date = d.timestamp.toDate ? d.timestamp.toDate() : new Date(d.timestamp);
    if (date < cutoff) return;
    const key = date.toLocaleDateString('vi-VN', { month: 'short', day: 'numeric' });
    if (key in counts) counts[key]++;
  });
  return counts;
});

const hasData = computed(() => props.docs.length > 0);

const lineData = computed(() => ({
  labels: dateLabels.value,
  datasets: [{
    label: 'Số phát hiện',
    data: dateLabels.value.map(l => dateCounts.value[l] ?? 0),
    borderColor: '#667eea',
    backgroundColor: 'rgba(102, 126, 234, 0.12)',
    pointBackgroundColor: '#667eea',
    pointRadius: 4,
    pointHoverRadius: 7,
    borderWidth: 2.5,
    fill: true,
    tension: 0.4,
  }],
}));

const lineOpts = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: { display: false },
    tooltip: { callbacks: { label: (ctx) => ` ${ctx.parsed.y} lần` } },
  },
  scales: {
    x: {
      grid: { display: false },
      ticks: { font: { size: 10 }, color: '#718096' },
    },
    y: {
      beginAtZero: true,
      ticks: { stepSize: 1, font: { size: 10 }, color: '#718096' },
      grid: { color: 'rgba(0,0,0,0.04)' },
    },
  },
};
</script>

<style scoped>
.chart-card {
  background: #fff;
  border: 1px solid #e2e8f0;
  border-radius: 14px;
  padding: 20px 24px;
  box-shadow: 0 1px 3px rgba(0,0,0,.05);
}

.chart-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 14px;
  padding-bottom: 10px;
  border-bottom: 1px solid #f0f0f0;
}

.chart-title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.88rem;
  font-weight: 700;
  color: #2d3748;
  margin: 0;
}

.period-toggle {
  display: flex;
  gap: 4px;
}
.period-toggle button {
  padding: 4px 12px;
  font-size: 0.75rem;
  font-weight: 600;
  border-radius: 6px;
  border: 1px solid #e2e8f0;
  background: #f8fafc;
  color: #64748b;
  cursor: pointer;
  transition: all 0.15s;
}
.period-toggle button.active {
  background: #667eea;
  color: #fff;
  border-color: #667eea;
}
.period-toggle button:hover:not(.active) { background: #edf2f7; }

.chart-wrap { height: 240px; position: relative; }

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
