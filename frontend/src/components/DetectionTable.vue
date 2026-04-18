<template>
  <div class="detection-table-wrap">
    <div class="table-header">
      <h4 class="table-title">
        <vue-feather type="list" size="16" />
        Nhật ký Phát hiện
        <span v-if="docs.length" class="count-chip">{{ docs.length }}</span>
      </h4>
      <!-- Filter: all / needs maintenance -->
      <div class="filter-row">
        <button :class="{ active: filter === 'all' }" @click="filter = 'all'">Tất cả</button>
        <button :class="{ active: filter === 'maintenance' }" @click="filter = 'maintenance'">
          <vue-feather type="tool" size="12" /> Cần bảo trì
        </button>
        <button :class="{ active: filter === 'valid' }" @click="filter = 'valid'">✅ Hợp lệ</button>
      </div>
    </div>

    <div class="table-container">
      <table class="dt" v-if="filtered.length > 0">
        <thead>
          <tr>
            <th>Ảnh</th>
            <th>Biển báo</th>
            <th>Độ chính xác</th>
            <th>Trạng thái</th>
            <th>Mô hình</th>
            <th>Thời gian</th>
            <th>Phản hồi</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="doc in filtered"
            :key="doc.id"
            :class="{ 'row-maintenance': maintenanceFlag(doc) }"
          >
            <!-- Thumbnail -->
            <td class="td-thumb">
              <div class="thumb-placeholder">
                <vue-feather type="image" size="18" />
              </div>
            </td>

            <!-- Sign name -->
            <td class="td-label">
              <span class="label-text">{{ doc.label || '—' }}</span>
            </td>

            <!-- Confidence -->
            <td class="td-conf">
              <div class="conf-bar-row">
                <div class="mini-bar">
                  <div
                    class="mini-fill"
                    :class="confClass(doc.confidence)"
                    :style="{ width: confPct(doc.confidence) + '%' }"
                  ></div>
                </div>
                <span :class="['conf-pct', confClass(doc.confidence)]">
                  {{ confPct(doc.confidence) }}%
                </span>
              </div>
            </td>

            <!-- Status -->
            <td class="td-status">
              <span v-if="maintenanceFlag(doc)" class="badge badge-maintenance">
                <vue-feather type="tool" size="11" /> Cần bảo trì
              </span>
              <span v-else-if="doc.is_valid" class="badge badge-valid">
                ✅ Hợp lệ
              </span>
              <span v-else class="badge badge-invalid">
                ❌ Không hợp lệ
              </span>
            </td>

            <!-- Model -->
            <td class="td-model">
              <span class="model-tag">{{ doc.model_used || 'N/A' }}</span>
            </td>

            <!-- Timestamp -->
            <td class="td-time">{{ formatTime(doc.timestamp) }}</td>

            <!-- Feedback buttons (Data-Flywheel placeholder) -->
            <td class="td-actions">
              <button
                class="action-btn confirm"
                :title="confirmed.has(doc.id) ? 'Đã xác nhận' : 'Xác nhận đúng'"
                @click="toggleConfirm(doc)"
                :disabled="reported.has(doc.id)"
              >
                <vue-feather :type="confirmed.has(doc.id) ? 'check-circle' : 'check'" size="13" />
              </button>
              <button
                class="action-btn report"
                :title="reported.has(doc.id) ? 'Đã báo lỗi' : 'Báo lỗi'"
                @click="toggleReport(doc)"
                :disabled="confirmed.has(doc.id)"
              >
                <vue-feather :type="reported.has(doc.id) ? 'x-circle' : 'x'" size="13" />
              </button>
            </td>
          </tr>
        </tbody>
      </table>

      <!-- Empty state -->
      <div v-else class="empty-table">
        <vue-feather type="inbox" size="36" />
        <p>Không có dữ liệu phù hợp</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue';
import VueFeather from 'vue-feather';

const props = defineProps({
  docs: { type: Array, default: () => [] },
});

const filter    = ref('all');
const confirmed = ref(new Set());
const reported  = ref(new Set());

// Helpers
const confPct   = (c) => ((c || 0) * 100).toFixed(0);
const confClass = (c) => (c || 0) < 0.5 ? 'low' : (c || 0) < 0.75 ? 'mid' : 'high';
const maintenanceFlag = (doc) => (doc.confidence || 0) < 0.5;

function formatTime(ts) {
  if (!ts) return '—';
  const d = ts.toDate ? ts.toDate() : new Date(ts);
  return d.toLocaleString('vi-VN', {
    day: '2-digit', month: '2-digit',
    hour: '2-digit', minute: '2-digit',
  });
}

const filtered = computed(() => {
  if (filter.value === 'maintenance') return props.docs.filter(maintenanceFlag);
  if (filter.value === 'valid')       return props.docs.filter(d => d.is_valid);
  return props.docs;
});

// Data-Flywheel feedback (local state for now — hook to Firestore later)
function toggleConfirm(doc) {
  const s = new Set(confirmed.value);
  s.has(doc.id) ? s.delete(doc.id) : s.add(doc.id);
  confirmed.value = s;
}
function toggleReport(doc) {
  const s = new Set(reported.value);
  s.has(doc.id) ? s.delete(doc.id) : s.add(doc.id);
  reported.value = s;
}
</script>

<style scoped>
.detection-table-wrap {
  background: #fff;
  border: 1px solid #e2e8f0;
  border-radius: 14px;
  overflow: hidden;
  box-shadow: 0 1px 3px rgba(0,0,0,.05);
}

.table-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 20px;
  border-bottom: 1px solid #f0f0f0;
  flex-wrap: wrap;
  gap: 10px;
}

.table-title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.9rem;
  font-weight: 700;
  color: #2d3748;
  margin: 0;
}

.count-chip {
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: #fff;
  font-size: 0.7rem;
  font-weight: 700;
  padding: 2px 8px;
  border-radius: 10px;
}

.filter-row {
  display: flex;
  gap: 6px;
}
.filter-row button {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 5px 12px;
  font-size: 0.76rem;
  font-weight: 600;
  border-radius: 6px;
  border: 1px solid #e2e8f0;
  background: #f8fafc;
  color: #64748b;
  cursor: pointer;
  transition: all 0.15s;
  white-space: nowrap;
}
.filter-row button.active   { background: #667eea; color: #fff; border-color: #667eea; }
.filter-row button:hover:not(.active) { background: #edf2f7; }

.table-container { overflow-x: auto; }

.dt { width: 100%; border-collapse: collapse; font-size: 0.84rem; }
.dt thead th {
  padding: 10px 14px;
  font-size: 0.72rem;
  font-weight: 700;
  color: #94a3b8;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  text-align: left;
  background: #f8fafc;
  border-bottom: 1px solid #e2e8f0;
  white-space: nowrap;
}
.dt tbody tr {
  border-bottom: 1px solid #f0f4f8;
  transition: background 0.15s;
}
.dt tbody tr:hover { background: #fafbff; }
.dt tbody tr:last-child { border-bottom: none; }
.dt tbody td { padding: 10px 14px; vertical-align: middle; }
.dt tbody tr.row-maintenance { background: #fffbf2; }
.dt tbody tr.row-maintenance:hover { background: #fff8eb; }

/* Cells */
.td-thumb .thumb-placeholder {
  width: 48px; height: 36px;
  background: #f0f4f8;
  border-radius: 6px;
  display: flex; align-items: center; justify-content: center;
  color: #94a3b8;
}

.label-text { font-weight: 600; color: #1a202c; }

.conf-bar-row { display: flex; align-items: center; gap: 8px; }
.mini-bar {
  width: 56px; height: 5px;
  background: #f0f0f0;
  border-radius: 3px;
  overflow: hidden;
  flex-shrink: 0;
}
.mini-fill { height: 100%; border-radius: 3px; transition: width 0.4s; }
.mini-fill.low  { background: #ef4444; }
.mini-fill.mid  { background: #f59e0b; }
.mini-fill.high { background: #22c55e; }
.conf-pct { font-weight: 700; font-size: 0.8rem; }
.conf-pct.low  { color: #ef4444; }
.conf-pct.mid  { color: #d97706; }
.conf-pct.high { color: #16a34a; }

.badge {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 4px 9px;
  border-radius: 6px;
  font-size: 0.72rem;
  font-weight: 700;
  white-space: nowrap;
}
.badge-maintenance { background: rgba(245,158,11,.12); color: #b45309; border: 1px solid rgba(245,158,11,.3); }
.badge-valid       { background: rgba(34,197,94,.1);  color: #15803d; border: 1px solid rgba(34,197,94,.3); }
.badge-invalid     { background: rgba(239,68,68,.08); color: #dc2626; border: 1px solid rgba(239,68,68,.2); }

.model-tag {
  font-size: 0.71rem;
  font-weight: 600;
  color: #764ba2;
  background: rgba(118,75,162,.08);
  padding: 3px 8px;
  border-radius: 6px;
  white-space: nowrap;
}

.td-time { color: #94a3b8; font-size: 0.78rem; white-space: nowrap; }

.td-actions { display: flex; gap: 6px; }
.action-btn {
  width: 28px; height: 28px;
  border-radius: 6px;
  border: 1px solid #e2e8f0;
  background: #f8fafc;
  display: flex; align-items: center; justify-content: center;
  cursor: pointer;
  transition: all 0.15s;
  color: #64748b;
}
.action-btn:hover:not(:disabled) { transform: scale(1.1); }
.action-btn.confirm:hover:not(:disabled) { background: rgba(34,197,94,.1); color: #16a34a; border-color: #22c55e; }
.action-btn.report:hover:not(:disabled)  { background: rgba(239,68,68,.08); color: #dc2626; border-color: #ef4444; }
.action-btn:disabled { opacity: 0.4; cursor: not-allowed; }

.empty-table {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  padding: 48px;
  color: #a0aec0;
}
.empty-table p { font-size: 0.88rem; }
</style>
