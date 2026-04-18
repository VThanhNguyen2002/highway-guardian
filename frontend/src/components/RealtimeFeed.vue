<template>
  <section class="panel realtime-feed">
    <h3>
      <vue-feather type="activity" size="18" />
      Lịch sử Thời gian thực
      <span v-if="docs.length > 0" class="count-badge">{{ docs.length }}</span>
    </h3>

    <!-- Firestore error -->
    <div v-if="firestoreError" class="feed-error">
      <vue-feather type="alert-triangle" size="16" />
      <div>
        <strong>Không thể tải dữ liệu</strong>
        <p>{{ firestoreError }}</p>
      </div>
    </div>

    <!-- Loading skeleton -->
    <div v-else-if="loading" class="feed-loading">
      <div v-for="i in 3" :key="i" class="skeleton-card"></div>
    </div>

    <!-- Empty state -->
    <div v-else-if="docs.length === 0" class="feedback-state idle">
      <vue-feather type="clock" size="32" class="idle-icon" />
      <p>Chưa có bản ghi. Hãy chạy nhận diện để bắt đầu.</p>
    </div>

    <!-- Feed cards -->
    <ul v-else class="feed-list">
      <li
        v-for="(doc, idx) in docs"
        :key="doc.id"
        class="feed-item"
        :class="{ 'slide-in': idx === 0 }"
      >
        <div class="feed-item-top">
          <span class="feed-label">{{ doc.label }}</span>
          <div class="feed-actions">
            <span class="feed-conf">{{ (doc.confidence * 100).toFixed(0) }}%</span>
            <button
              class="action-btn edit-btn"
              title="Sửa nhãn"
              @click="openEdit(doc)"
            >
              <vue-feather type="edit-2" size="13" />
            </button>
            <button
              class="action-btn delete-btn"
              title="Xoá (False Positive)"
              @click="confirmDelete(doc)"
            >
              <vue-feather type="trash-2" size="13" />
            </button>
          </div>
        </div>
        <div class="feed-item-meta">
          <span class="feed-validity" :class="doc.is_valid ? 'valid' : 'invalid'">
            {{ doc.is_valid ? '✅ Hợp lệ' : '❌ Không hợp lệ' }}
          </span>
          <span class="feed-sep">·</span>
          <vue-feather type="cpu" size="11" />
          <span class="feed-model">{{ doc.model_used || 'Ensemble' }}</span>
          <span class="feed-sep">·</span>
          <vue-feather type="user" size="11" />
          {{ doc.performed_by }}
          <span class="feed-time">{{ formatTime(doc.timestamp) }}</span>
        </div>
        <div v-if="doc.box_coordinates && doc.box_coordinates.length === 4" class="feed-bbox">
          <vue-feather type="maximize-2" size="11" />
          [{{ doc.box_coordinates.map(v => Math.round(v)).join(', ') }}]
        </div>
      </li>
    </ul>

    <!-- Live indicator -->
    <div v-if="isListening && !firestoreError" class="live-indicator">
      <span class="live-dot"></span> Đang lắng nghe Firestore
    </div>

    <!-- ── Edit Label Modal ─────────────────────────────────────────── -->
    <transition name="modal-fade">
      <div v-if="editTarget" class="modal-overlay" @click.self="closeEdit">
        <div class="modal-box">
          <div class="modal-header">
            <vue-feather type="edit-2" size="16" />
            <h4>Sửa nhãn phát hiện</h4>
            <button class="modal-close" @click="closeEdit">
              <vue-feather type="x" size="16" />
            </button>
          </div>
          <p class="modal-hint">Nhãn hiện tại: <strong>{{ editTarget.label }}</strong></p>
          <select v-model="editNewLabel" class="label-select">
            <option v-for="cls in ZALO_CLASSES" :key="cls" :value="cls">{{ cls }}</option>
          </select>
          <div class="modal-footer">
            <button class="btn btn-secondary" @click="closeEdit">Huỷ</button>
            <button class="btn btn-primary" :disabled="saving" @click="saveEdit">
              <vue-feather v-if="saving" type="loader" size="14" class="spin" />
              {{ saving ? 'Đang lưu…' : 'Lưu' }}
            </button>
          </div>
        </div>
      </div>
    </transition>
  </section>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue';
import VueFeather from 'vue-feather';
import { db } from '../firebase/config';
import {
  collection, onSnapshot, orderBy, query, limit,
  deleteDoc, updateDoc, doc
} from 'firebase/firestore';

// ── Zalo AI 2020 classes (index 1–7) ──────────────────────────────────────
const ZALO_CLASSES = [
  'Biển cấm',
  'Giới hạn tốc độ',
  'Cấm vượt',
  'Hướng đi',
  'Biển hiệu lệnh',
  'Biển cảnh báo',
  'Biển phụ',
];

const docs          = ref([]);
const loading       = ref(true);
const isListening   = ref(false);
const firestoreError = ref(null);

// Edit state
const editTarget  = ref(null);
const editNewLabel = ref('');
const saving      = ref(false);

let unsubscribe = null;

function formatTime(ts) {
  if (!ts) return '';
  const date = ts.toDate ? ts.toDate() : new Date(ts);
  return date.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

// ── CRUD helpers ────────────────────────────────────────────────────────────

async function confirmDelete(document) {
  if (!confirm(`Xoá kết quả phát hiện "${document.label}"? Hành động này không thể hoàn tác.`)) return;
  try {
    await deleteDoc(doc(db, 'detections', document.id));
  } catch (e) {
    alert(`Xoá thất bại: ${e.message}`);
    console.error('[RealtimeFeed] deleteDoc error:', e);
  }
}

function openEdit(document) {
  editTarget.value  = document;
  editNewLabel.value = ZALO_CLASSES.includes(document.label) ? document.label : ZALO_CLASSES[0];
}

function closeEdit() {
  editTarget.value  = null;
  editNewLabel.value = '';
}

async function saveEdit() {
  if (!editTarget.value || !editNewLabel.value) return;
  saving.value = true;
  try {
    await updateDoc(doc(db, 'detections', editTarget.value.id), {
      label: editNewLabel.value,
      corrected: true,
      corrected_at: new Date(),
    });
    closeEdit();
  } catch (e) {
    alert(`Cập nhật thất bại: ${e.message}`);
    console.error('[RealtimeFeed] updateDoc error:', e);
  } finally {
    saving.value = false;
  }
}

// ── Firestore listener ───────────────────────────────────────────────────────

function startListener() {
  try {
    const q = query(
      collection(db, 'detections'),
      orderBy('timestamp', 'desc'),
      limit(30)
    );
    isListening.value = true;
    unsubscribe = onSnapshot(
      q,
      (snap) => {
        loading.value        = false;
        firestoreError.value = null;
        docs.value = snap.docs.map(d => ({ id: d.id, ...d.data() }));
      },
      (err) => {
        loading.value       = false;
        isListening.value   = false;
        if (err.code === 'failed-precondition') {
          firestoreError.value = 'Cần tạo composite index trong Firestore. Kiểm tra console để biết link tạo index.';
        } else if (err.code === 'permission-denied') {
          firestoreError.value = 'Bị từ chối quyền truy cập Firestore. Kiểm tra Security Rules.';
        } else {
          firestoreError.value = `Lỗi Firestore: ${err.message}`;
        }
        console.error('[RealtimeFeed] Firestore error:', err);
      }
    );
  } catch (e) {
    loading.value        = false;
    isListening.value    = false;
    firestoreError.value = `Không thể kết nối Firestore: ${e.message}`;
    console.error('[RealtimeFeed] Failed to start listener:', e);
  }
}

onMounted(startListener);
onUnmounted(() => {
  if (unsubscribe) { unsubscribe(); unsubscribe = null; }
  isListening.value = false;
});
</script>

<style scoped>
.realtime-feed { display: flex; flex-direction: column; gap: 16px; min-height: 400px; }
.realtime-feed h3 { display: flex; align-items: center; gap: 10px; }

.count-badge {
  margin-left: auto;
  font-size: 0.72rem;
  font-weight: 700;
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: #fff;
  padding: 2px 9px;
  border-radius: 10px;
}

/* Error */
.feed-error {
  display: flex; align-items: flex-start; gap: 12px;
  padding: 14px 16px;
  background: rgba(239,68,68,0.06);
  border: 1px solid rgba(239,68,68,0.3);
  border-left: 4px solid #ef4444;
  border-radius: 8px; color: #991b1b; font-size: 0.85rem;
}
.feed-error strong { display: block; margin-bottom: 2px; }
.feed-error p      { margin: 0; color: #b91c1c; }

/* Skeleton */
.feed-loading { display: flex; flex-direction: column; gap: 8px; }
.skeleton-card {
  height: 72px; border-radius: 8px;
  background: linear-gradient(90deg, #f0f0f0 25%, #e8e8e8 50%, #f0f0f0 75%);
  background-size: 200% 100%;
  animation: shimmer 1.4s infinite;
}
@keyframes shimmer {
  0%   { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}

/* Feed list */
.feed-list {
  list-style: none; padding: 0; margin: 0;
  display: flex; flex-direction: column; gap: 8px;
  max-height: calc(100vh - 320px);
  overflow-y: auto; padding-right: 4px;
}
.feed-list::-webkit-scrollbar       { width: 5px; }
.feed-list::-webkit-scrollbar-track { background: transparent; }
.feed-list::-webkit-scrollbar-thumb { background: #cbd5e0; border-radius: 3px; }

.feed-item {
  padding: 12px 16px;
  background: #fff;
  border: 1px solid #e2e8f0;
  border-left: 4px solid #22c55e;
  border-radius: 8px;
  transition: box-shadow 0.2s, border-left-color 0.2s;
}
.feed-item:hover { border-left-color: #16a34a; box-shadow: 0 2px 8px rgba(34,197,94,0.1); }
.feed-item.slide-in { animation: slideIn 0.3s ease-out; }
@keyframes slideIn {
  from { opacity: 0; transform: translateY(-10px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* Top row */
.feed-item-top {
  display: flex; align-items: center;
  justify-content: space-between; margin-bottom: 6px;
}
.feed-label { font-weight: 700; color: #1a202c; font-size: 0.95rem; }

/* Actions */
.feed-actions { display: flex; align-items: center; gap: 6px; }

.feed-conf {
  font-weight: 700; color: #22c55e;
  background: rgba(34,197,94,0.1);
  padding: 2px 8px; border-radius: 8px; font-size: 0.8rem;
}

.action-btn {
  display: flex; align-items: center; justify-content: center;
  width: 26px; height: 26px; border-radius: 6px; border: none;
  cursor: pointer; transition: all 0.18s; flex-shrink: 0;
}
.edit-btn   { background: rgba(102,126,234,0.1); color: #667eea; }
.edit-btn:hover   { background: #667eea; color: #fff; transform: scale(1.08); }
.delete-btn { background: rgba(239,68,68,0.08); color: #ef4444; }
.delete-btn:hover { background: #ef4444; color: #fff; transform: scale(1.08); }

/* Meta row */
.feed-item-meta {
  display: flex; align-items: center; gap: 5px;
  color: #94a3b8; font-size: 0.77rem; flex-wrap: wrap;
}
.feed-sep      { color: #cbd5e0; }
.feed-model    { font-weight: 600; color: #667eea; }
.feed-validity { font-weight: 600; font-size: 0.75rem; }
.feed-validity.valid   { color: #16a34a; }
.feed-validity.invalid { color: #dc2626; }
.feed-time { margin-left: auto; font-variant-numeric: tabular-nums; color: #a0aec0; }

.feed-bbox {
  margin-top: 5px; display: flex; align-items: center; gap: 4px;
  color: #94a3b8; font-size: 0.72rem; font-family: monospace;
}

.feedback-state {
  display: flex; align-items: center; justify-content: center;
  flex-direction: column; gap: 10px; padding: 28px;
  border: 2px dashed #e2e8f0; border-radius: 10px;
  background: rgba(102,126,234,.02); color: #718096;
  font-weight: 500; font-size: 0.9rem; text-align: center;
}
.feedback-state.idle { color: #a0aec0; font-style: italic; }
.idle-icon { color: #cbd5e0; margin-bottom: 4px; }

/* Live indicator */
.live-indicator {
  display: flex; align-items: center; gap: 6px;
  font-size: 0.75rem; font-weight: 600; color: #059669; padding-top: 4px;
}
.live-dot {
  width: 7px; height: 7px; border-radius: 50%; background: #10b981;
  animation: pulse-dot 1.5s ease-in-out infinite;
}
@keyframes pulse-dot {
  0%, 100% { opacity: 1; transform: scale(1); }
  50%       { opacity: 0.5; transform: scale(1.4); }
}

/* ── Modal ──────────────────────────────────────────────────────────────── */
.modal-overlay {
  position: fixed; inset: 0; z-index: 1000;
  background: rgba(10,10,20,0.55);
  backdrop-filter: blur(4px);
  display: flex; align-items: center; justify-content: center;
}
.modal-box {
  background: #fff; border-radius: 14px;
  padding: 28px 32px; width: 380px; max-width: 94vw;
  box-shadow: 0 20px 60px rgba(0,0,0,0.22);
  display: flex; flex-direction: column; gap: 16px;
}
.modal-header {
  display: flex; align-items: center; gap: 10px; color: #1a202c;
}
.modal-header h4 { margin: 0; flex: 1; font-size: 1rem; font-weight: 700; }
.modal-close {
  background: none; border: none; cursor: pointer;
  color: #94a3b8; padding: 4px; border-radius: 6px;
  transition: color 0.15s, background 0.15s;
}
.modal-close:hover { color: #ef4444; background: rgba(239,68,68,0.08); }

.modal-hint { margin: 0; color: #64748b; font-size: 0.875rem; }
.modal-hint strong { color: #1a202c; }

.label-select {
  width: 100%; padding: 9px 12px; border-radius: 8px;
  border: 1.5px solid #e2e8f0; font-size: 0.9rem; color: #1a202c;
  background: #f8fafc; cursor: pointer;
  transition: border-color 0.2s;
  outline: none;
}
.label-select:focus { border-color: #667eea; }

.modal-footer { display: flex; justify-content: flex-end; gap: 10px; }

.btn {
  padding: 8px 20px; border-radius: 8px; font-size: 0.875rem;
  font-weight: 600; cursor: pointer; border: none;
  display: flex; align-items: center; gap: 6px;
  transition: all 0.18s;
}
.btn-secondary { background: #f1f5f9; color: #475569; }
.btn-secondary:hover { background: #e2e8f0; }
.btn-primary {
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: #fff;
}
.btn-primary:hover:not(:disabled) { opacity: 0.9; transform: translateY(-1px); }
.btn-primary:disabled { opacity: 0.55; cursor: not-allowed; }

.spin { animation: spin-icon 0.8s linear infinite; }
@keyframes spin-icon {
  to { transform: rotate(360deg); }
}

/* Modal transition */
.modal-fade-enter-active, .modal-fade-leave-active { transition: opacity 0.2s; }
.modal-fade-enter-from, .modal-fade-leave-to { opacity: 0; }
</style>
