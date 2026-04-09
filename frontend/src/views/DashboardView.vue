<template>
  <div class="dashboard-view">
    <!-- Header -->
    <div class="dash-header">
      <div>
        <h1>🛡️ Highway Guardian Analytics</h1>
        <p class="subtitle">Real-time traffic sign compliance dashboard</p>
      </div>
      <div class="dash-controls">
        <label>Lookback</label>
        <select v-model.number="store.lookbackDays" @change="store.loadAll()">
          <option :value="7">7 days</option>
          <option :value="30">30 days</option>
          <option :value="90">90 days</option>
          <option :value="365">1 year</option>
        </select>
        <button class="btn-refresh" :disabled="store.loading" @click="store.loadAll()">
          {{ store.loading ? '⏳ Loading…' : '🔄 Refresh' }}
        </button>
      </div>
    </div>

    <!-- Error banner -->
    <div v-if="store.error" class="error-banner">
      ⚠️ {{ store.error }}
    </div>

    <!-- Empty state -->
    <div v-if="!store.loading && !store.error && (!store.validity || store.validity.total === 0)" class="empty-state-card">
      <div class="empty-icon">📊</div>
      <h2>No Analytics Available</h2>
      <p>The dashboard is empty. Upload images in the Detect interface to generate statistics.</p>
    </div>

    <!-- KPI Cards -->
    <div class="kpi-row" v-if="store.validity && store.validity.total > 0">
      <div class="kpi-card">
        <span class="kpi-label">Total Detections</span>
        <span class="kpi-value">{{ store.validity.total.toLocaleString() }}</span>
      </div>
      <div class="kpi-card kpi-valid">
        <span class="kpi-label">Valid Signs</span>
        <span class="kpi-value">{{ store.validity.valid_count.toLocaleString() }}</span>
      </div>
      <div class="kpi-card kpi-invalid">
        <span class="kpi-label">Invalid Signs</span>
        <span class="kpi-value">{{ store.validity.invalid_count.toLocaleString() }}</span>
      </div>
      <div class="kpi-card kpi-ratio">
        <span class="kpi-label">Pass Rate</span>
        <span class="kpi-value">{{ (store.validity.valid_ratio * 100).toFixed(1) }}%</span>
      </div>
    </div>

    <!-- Chart Grid -->
    <div class="chart-grid" v-if="store.validity && store.validity.total > 0">
      <div class="chart-wide">
        <TrendChart />
      </div>
      <div class="chart-half">
        <ValidityPieChart />
      </div>
      <div class="chart-half" style="height: 380px">
        <SignFrequencyChart />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted } from 'vue'
import { useAnalyticsStore } from '@/stores/analytics'
import TrendChart from '@/components/charts/TrendChart.vue'
import ValidityPieChart from '@/components/charts/ValidityPieChart.vue'
import SignFrequencyChart from '@/components/charts/SignFrequencyChart.vue'

const store = useAnalyticsStore()

onMounted(() => {
  store.loadAll()
})
</script>

<style scoped>
.dashboard-view {
  padding: 1.5rem 2rem;
  max-width: 1400px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.dash-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  flex-wrap: wrap;
  gap: 1rem;
}
.dash-header h1 {
  font-size: 1.8rem;
  font-weight: 700;
  margin: 0;
}
.subtitle {
  color: #94a3b8;
  font-size: 0.9rem;
  margin: 0.25rem 0 0;
}
.dash-controls {
  display: flex;
  align-items: center;
  gap: 0.6rem;
}
.dash-controls label {
  font-size: 0.85rem;
  color: #94a3b8;
}
.dash-controls select {
  background: #1e293b;
  border: 1px solid #334155;
  color: #e2e8f0;
  border-radius: 6px;
  padding: 0.35rem 0.6rem;
  font-size: 0.85rem;
}
.btn-refresh {
  background: #6366f1;
  color: white;
  border: none;
  border-radius: 6px;
  padding: 0.4rem 0.9rem;
  font-size: 0.85rem;
  cursor: pointer;
  transition: background 0.2s;
}
.btn-refresh:hover:not(:disabled) { background: #4f46e5; }
.btn-refresh:disabled { opacity: 0.5; cursor: not-allowed; }

.error-banner {
  background: #450a0a;
  border: 1px solid #ef4444;
  border-radius: 8px;
  padding: 0.75rem 1rem;
  color: #fca5a5;
  font-size: 0.9rem;
}

/* KPI row */
.kpi-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1rem;
}
@media (max-width: 768px) {
  .kpi-row { grid-template-columns: repeat(2, 1fr); }
}
.kpi-card {
  background: #1e293b;
  border: 1px solid #334155;
  border-radius: 12px;
  padding: 1.2rem 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}
.kpi-label { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }
.kpi-value { font-size: 2rem; font-weight: 700; color: #e2e8f0; }
.kpi-valid .kpi-value { color: #22c55e; }
.kpi-invalid .kpi-value { color: #ef4444; }
.kpi-ratio .kpi-value { color: #6366f1; }

/* Chart grid */
.chart-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
}
.chart-wide {
  grid-column: 1 / -1;
  height: 300px;
}
.chart-half { min-height: 300px; }
@media (max-width: 900px) {
  .chart-grid { grid-template-columns: 1fr; }
  .chart-wide { grid-column: 1; }
}

/* Chart card base style (referenced by child components) */
:deep(.chart-card) {
  background: #1e293b;
  border: 1px solid #334155;
  border-radius: 12px;
  padding: 1.25rem;
  height: 100%;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}
:deep(.chart-header) {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
:deep(.chart-header h3) {
  font-size: 1rem;
  font-weight: 600;
  margin: 0;
  color: #e2e8f0;
}
:deep(.btn-toggle) {
  background: transparent;
  border: 1px solid #334155;
  color: #94a3b8;
  border-radius: 5px;
  padding: 0.25rem 0.6rem;
  font-size: 0.78rem;
  cursor: pointer;
  margin-left: 4px;
  transition: all 0.15s;
}
:deep(.btn-toggle.active),
:deep(.btn-toggle:hover) {
  background: #6366f1;
  color: white;
  border-color: #6366f1;
}
:deep(.chart-placeholder) {
  display: flex;
  align-items: center;
  justify-content: center;
  flex: 1;
  color: #475569;
  font-size: 0.9rem;
}

.empty-state-card {
  background: #1e293b;
  border: 1px dashed #334155;
  border-radius: 12px;
  padding: 4rem 2rem;
  text-align: center;
  color: #94a3b8;
}
.empty-icon {
  font-size: 3rem;
  margin-bottom: 1rem;
}
.empty-state-card h2 {
  color: #e2e8f0;
  margin-bottom: 0.5rem;
}
</style>
