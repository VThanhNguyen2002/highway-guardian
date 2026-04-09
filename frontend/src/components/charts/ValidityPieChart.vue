<template>
  <!-- Valid vs Invalid Doughnut Chart -->
  <div class="chart-card">
    <div class="chart-header">
      <h3>✅ Validity Ratio</h3>
    </div>
    <Doughnut v-if="chartData" :data="chartData" :options="chartOptions" />
    <div v-else class="chart-placeholder">
      <span>No validity data available.</span>
    </div>
    <div v-if="store.validity" class="validity-stats">
      <span class="stat valid">✅ {{ store.validity.valid_count }} Valid</span>
      <span class="stat invalid">❌ {{ store.validity.invalid_count }} Invalid</span>
      <span class="stat ratio">{{ (store.validity.valid_ratio * 100).toFixed(1) }}% pass rate</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { ArcElement, Chart as ChartJS, Legend, Tooltip } from 'chart.js'
import { Doughnut } from 'vue-chartjs'
import { useAnalyticsStore } from '@/stores/analytics'

ChartJS.register(ArcElement, Tooltip, Legend)

const store = useAnalyticsStore()

const chartData = computed(() => {
  if (!store.validity) return null
  return {
    labels: ['Valid', 'Invalid'],
    datasets: [
      {
        data: [store.validity.valid_count, store.validity.invalid_count],
        backgroundColor: ['#22c55e', '#ef4444'],
        borderColor: ['#16a34a', '#dc2626'],
        borderWidth: 2,
        hoverOffset: 6,
      },
    ],
  }
})

const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  cutout: '68%',
  plugins: {
    legend: { position: 'bottom' as const },
    tooltip: {
      callbacks: {
        label: (ctx: { label: string; raw: number }) =>
          ` ${ctx.label}: ${ctx.raw} detections`,
      },
    },
  },
}
</script>

<style scoped>
.validity-stats {
  display: flex;
  justify-content: center;
  gap: 1.2rem;
  margin-top: 0.75rem;
  font-size: 0.85rem;
  flex-wrap: wrap;
}
.stat { font-weight: 600; }
.stat.valid { color: #22c55e; }
.stat.invalid { color: #ef4444; }
.stat.ratio { color: #6366f1; }
</style>
