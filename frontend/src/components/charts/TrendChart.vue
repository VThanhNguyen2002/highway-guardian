<template>
  <!-- Detection Trend — Line Chart (daily or monthly) -->
  <div class="chart-card">
    <div class="chart-header">
      <h3>📈 Detection Trend</h3>
      <div class="chart-controls">
        <button
          v-for="g in ['daily', 'monthly'] as const"
          :key="g"
          :class="['btn-toggle', { active: store.granularity === g }]"
          @click="onGranularityChange(g)"
        >
          {{ g === 'daily' ? 'Daily' : 'Monthly' }}
        </button>
      </div>
    </div>
    <Line v-if="chartData" :data="chartData" :options="chartOptions" />
    <div v-else class="chart-placeholder">
      <span>No trend data available.</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, watch } from 'vue'
import {
  CategoryScale,
  Chart as ChartJS,
  Filler,
  Legend,
  LinearScale,
  LineElement,
  PointElement,
  Title,
  Tooltip,
} from 'chart.js'
import { Line } from 'vue-chartjs'
import { useAnalyticsStore } from '@/stores/analytics'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
)

const store = useAnalyticsStore()

const chartData = computed(() => {
  if (!store.trend?.data.length) return null
  return {
    labels: store.trend.data.map((p) => p.period),
    datasets: [
      {
        label: 'Detections',
        data: store.trend.data.map((p) => p.count),
        borderColor: '#6366f1',
        backgroundColor: 'rgba(99,102,241,0.12)',
        fill: true,
        tension: 0.4,
        pointRadius: 4,
        pointBackgroundColor: '#6366f1',
      },
    ],
  }
})

const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: { display: false },
    tooltip: {
      callbacks: {
        label: (ctx: { parsed: { y: number } }) => ` ${ctx.parsed.y} detections`,
      },
    },
  },
  scales: {
    y: { beginAtZero: true, ticks: { stepSize: 1 } },
  },
}

function onGranularityChange(g: 'daily' | 'monthly'): void {
  store.granularity = g
  store.loadAll()
}
</script>
