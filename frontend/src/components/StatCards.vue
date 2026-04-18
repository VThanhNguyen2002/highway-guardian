<template>
  <div class="stat-cards-grid">
    <!-- Total Scans -->
    <div class="stat-card stat-total">
      <div class="stat-icon">
        <vue-feather type="layers" size="22" />
      </div>
      <div class="stat-body">
        <span class="stat-label">Tổng quét</span>
        <span class="stat-value">{{ total.toLocaleString() }}</span>
        <span class="stat-sub">{{ todayCount }} hôm nay</span>
      </div>
      <div class="stat-trend up" v-if="todayCount > 0">
        <vue-feather type="trending-up" size="14" />
        Hoạt động
      </div>
    </div>

    <!-- AI Confidence -->
    <div class="stat-card stat-confidence">
      <div class="stat-icon">
        <vue-feather type="target" size="22" />
      </div>
      <div class="stat-body">
        <span class="stat-label">Độ tin cậy TB</span>
        <span class="stat-value">{{ avgConfidence }}<small>%</small></span>
        <span class="stat-sub">{{ total }} mẫu gần nhất</span>
      </div>
      <div class="conf-bar">
        <div class="conf-fill" :style="{ width: avgConfidenceRaw + '%' }"></div>
      </div>
    </div>

    <!-- Alert / Danger count -->
    <div class="stat-card stat-alert" :class="{ 'has-alerts': alertCount > 0 }">
      <div class="stat-icon">
        <vue-feather type="alert-triangle" size="22" />
      </div>
      <div class="stat-body">
        <span class="stat-label">Cảnh báo hôm nay</span>
        <span class="stat-value">{{ alertCount }}</span>
        <span class="stat-sub">Cấm / Nguy hiểm</span>
      </div>
      <div class="stat-trend" :class="alertCount > 0 ? 'warn' : 'ok'">
        <vue-feather :type="alertCount > 0 ? 'alert-circle' : 'check'" size="14" />
        {{ alertCount > 0 ? 'Cần chú ý' : 'Bình thường' }}
      </div>
    </div>

    <!-- Low Confidence / Maintenance -->
    <div class="stat-card stat-maintenance">
      <div class="stat-icon">
        <vue-feather type="tool" size="22" />
      </div>
      <div class="stat-body">
        <span class="stat-label">Cần bảo trì</span>
        <span class="stat-value">{{ maintenanceCount }}</span>
        <span class="stat-sub">Độ chính xác &lt;50%</span>
      </div>
      <div class="stat-trend" :class="maintenanceCount > 0 ? 'warn' : 'ok'">
        <vue-feather :type="maintenanceCount > 0 ? 'tool' : 'check'" size="14" />
        {{ maintenanceCount > 0 ? 'Có biển xuống cấp' : 'Tốt' }}
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue';
import VueFeather from 'vue-feather';

const props = defineProps({
  docs: { type: Array, default: () => [] },
});

// Zalo class IDs that correspond to Prohibited/Danger super-categories
const ALERT_CLASS_IDS = new Set([1, 2, 3, 4, 5, 6, 7]); // all Zalo classes are traffic-related

const total = computed(() => props.docs.length);

const todayCount = computed(() => {
  const today = new Date().toDateString();
  return props.docs.filter(d => {
    if (!d.timestamp) return false;
    const date = d.timestamp.toDate ? d.timestamp.toDate() : new Date(d.timestamp);
    return date.toDateString() === today;
  }).length;
});

const avgConfidenceRaw = computed(() => {
  if (!total.value) return 0;
  const sum = props.docs.reduce((a, d) => a + (d.confidence || 0), 0);
  return parseFloat(((sum / total.value) * 100).toFixed(1));
});

const avgConfidence = computed(() =>
  total.value === 0 ? '—' : avgConfidenceRaw.value.toFixed(1)
);

const alertCount = computed(() => {
  const today = new Date().toDateString();
  return props.docs.filter(d => {
    if (!d.timestamp) return false;
    const date = d.timestamp.toDate ? d.timestamp.toDate() : new Date(d.timestamp);
    return date.toDateString() === today && !d.is_valid;
  }).length;
});

const maintenanceCount = computed(() =>
  props.docs.filter(d => (d.confidence || 0) < 0.5).length
);
</script>

<style scoped>
.stat-cards-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 18px;
}
@media (max-width: 1100px) { .stat-cards-grid { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 600px)  { .stat-cards-grid { grid-template-columns: 1fr; } }

.stat-card {
  background: #fff;
  border: 1px solid #e2e8f0;
  border-radius: 14px;
  padding: 20px 22px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  box-shadow: 0 1px 4px rgba(0,0,0,.05);
  transition: box-shadow 0.2s, transform 0.2s;
  position: relative;
  overflow: hidden;
}
.stat-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
}
.stat-card:hover { box-shadow: 0 6px 16px rgba(0,0,0,.09); transform: translateY(-2px); }

.stat-total::before      { background: linear-gradient(90deg, #667eea, #764ba2); }
.stat-confidence::before { background: linear-gradient(90deg, #22c55e, #16a34a); }
.stat-alert::before      { background: linear-gradient(90deg, #f59e0b, #ef4444); }
.stat-maintenance::before{ background: linear-gradient(90deg, #06b6d4, #8b5cf6); }

.stat-card.has-alerts { border-color: #fca5a5; background: #fff8f8; }
.stat-card.has-alerts::before { background: linear-gradient(90deg, #ef4444, #dc2626); }

.stat-icon {
  width: 44px; height: 44px;
  border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
}
.stat-total .stat-icon      { background: rgba(102,126,234,.12); color: #667eea; }
.stat-confidence .stat-icon { background: rgba(34,197,94,.12);  color: #16a34a; }
.stat-alert .stat-icon      { background: rgba(245,158,11,.12); color: #d97706; }
.stat-maintenance .stat-icon{ background: rgba(6,182,212,.12);  color: #0891b2; }

.stat-body { display: flex; flex-direction: column; gap: 2px; }
.stat-label { font-size: 0.73rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }
.stat-value { font-size: 2rem; font-weight: 800; color: #1a202c; line-height: 1.1; }
.stat-value small { font-size: 1rem; font-weight: 600; color: #64748b; }
.stat-sub { font-size: 0.75rem; color: #94a3b8; font-weight: 500; }

.stat-trend {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 0.73rem;
  font-weight: 700;
  padding: 4px 10px;
  border-radius: 20px;
  width: fit-content;
}
.stat-trend.up   { background: rgba(34,197,94,.1);  color: #16a34a; }
.stat-trend.warn { background: rgba(245,158,11,.1); color: #d97706; }
.stat-trend.ok   { background: rgba(34,197,94,.1);  color: #16a34a; }

/* Confidence progress bar */
.conf-bar {
  height: 4px;
  background: #f0f0f0;
  border-radius: 2px;
  overflow: hidden;
  margin-top: 2px;
}
.conf-fill {
  height: 100%;
  background: linear-gradient(90deg, #22c55e, #16a34a);
  border-radius: 2px;
  transition: width 0.6s ease;
}
</style>
