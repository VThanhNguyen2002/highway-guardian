<template>
  <div class="analytics-orchestrator">
    <!-- Error banner -->
    <div v-if="error" class="analytics-error">
      <vue-feather type="alert-triangle" size="14" />
      {{ error }}
    </div>

    <!-- Loading -->
    <div v-else-if="loading" class="skeleton-grid">
      <div v-for="i in 4" :key="i" class="skeleton-kpi"></div>
    </div>

    <template v-else>
      <!-- KPI cards -->
      <StatCards :docs="allDocs" />

      <!-- Distribution + Time series -->
      <div class="chart-row">
        <DistributionCharts :docs="allDocs" />
      </div>
      <TimeSeriesCharts :docs="allDocs" />

      <!-- Logs table -->
      <DetectionTable :docs="allDocs" />
    </template>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue';
import VueFeather    from 'vue-feather';
import StatCards        from './StatCards.vue';
import DistributionCharts from './DistributionCharts.vue';
import TimeSeriesCharts   from './TimeSeriesCharts.vue';
import DetectionTable     from './DetectionTable.vue';
import { db } from '../firebase/config';
import { collection, onSnapshot, orderBy, query, limit } from 'firebase/firestore';

const allDocs = ref([]);
const loading = ref(true);
const error   = ref(null);
let   unsub   = null;

function startListener() {
  try {
    const q = query(collection(db, 'detections'), orderBy('timestamp', 'desc'), limit(200));
    unsub = onSnapshot(
      q,
      (snap) => {
        loading.value = false;
        error.value   = null;
        allDocs.value = snap.docs.map(d => ({ id: d.id, ...d.data() }));
      },
      (err) => {
        loading.value = false;
        if (err.code === 'failed-precondition') {
          error.value = 'Cần composite index trong Firestore (kiểm tra console để tạo index).';
        } else if (err.code === 'permission-denied') {
          error.value = 'Bị từ chối quyền truy cập Firestore. Kiểm tra Security Rules.';
        } else {
          error.value = `Lỗi Firestore: ${err.message}`;
        }
        console.error('[AnalyticsCharts] Firestore error:', err);
      }
    );
  } catch (e) {
    loading.value = false;
    error.value   = `Không thể kết nối: ${e.message}`;
  }
}

onMounted(startListener);
onUnmounted(() => { if (unsub) unsub(); });
</script>

<style scoped>
.analytics-orchestrator {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.analytics-error {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 16px;
  background: rgba(239,68,68,.06);
  border: 1px solid rgba(239,68,68,.3);
  border-left: 4px solid #ef4444;
  border-radius: 8px;
  color: #991b1b;
  font-size: 0.85rem;
  font-weight: 500;
}

.skeleton-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
}
.skeleton-kpi {
  height: 88px;
  border-radius: 14px;
  background: linear-gradient(90deg, #f0f0f0 25%, #e8e8e8 50%, #f0f0f0 75%);
  background-size: 200% 100%;
  animation: shimmer 1.4s infinite;
}
@keyframes shimmer {
  0%   { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}

.chart-row {
  /* DistributionCharts is self-contained grid, just pass through */
}
</style>
