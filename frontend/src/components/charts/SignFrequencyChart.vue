<template>
  <!-- Top-N Sign Category Frequency — Horizontal Bar Chart -->
  <div class="chart-card">
    <div class="chart-header">
      <h3>📊 Sign Frequency</h3>
    </div>
    <Bar v-if="chartData" :data="chartData" :options="chartOptions" />
    <div v-else class="chart-placeholder">
      <span>No frequency data available.</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import {
  BarElement,
  CategoryScale,
  Chart as ChartJS,
  Legend,
  LinearScale,
  Title,
  Tooltip,
} from 'chart.js'
import { Bar } from 'vue-chartjs'
import { useAnalyticsStore } from '@/stores/analytics'

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend)

const store = useAnalyticsStore()

const PALETTE = [
  '#6366f1', '#8b5cf6', '#ec4899', '#f43f5e', '#f97316',
  '#eab308', '#22c55e', '#14b8a6', '#06b6d4', '#3b82f6',
]

const chartData = computed(() => {
  if (!store.frequency?.data.length) return null
  const items = [...store.frequency.data].sort((a, b) => b.count - a.count)
  return {
    labels: items.map((p) => p.class_name),
    datasets: [
      {
        label: 'Detections',
        data: items.map((p) => p.count),
        backgroundColor: items.map((_, i) => PALETTE[i % PALETTE.length]),
        borderRadius: 6,
        borderSkipped: false,
      },
    ],
  }
})

const chartOptions = {
  indexAxis: 'y' as const,   // horizontal bars
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: { display: false },
    tooltip: {
      callbacks: {
        label: (ctx: { parsed: { x: number } }) => ` ${ctx.parsed.x} detections`,
      },
    },
  },
  scales: {
    x: { beginAtZero: true, ticks: { stepSize: 1 } },
    y: { ticks: { font: { size: 12 } } },
  },
}
</script>
