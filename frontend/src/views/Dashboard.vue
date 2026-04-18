<template>
  <main class="main-content dashboard-page">

    <!-- ── Page header ─────────────────────────────────────────────────── -->
    <div class="page-header">
      <div class="page-title">
        <h1>
          <vue-feather type="shield" size="24" class="title-icon" />
          Traffic Infrastructure Dashboard
        </h1>
        <p class="page-subtitle">
          Quản lý &amp; Phân tích hạ tầng biển báo giao thông
        </p>
      </div>
      <div class="header-actions">
        <span class="live-pill">
          <span class="live-dot"></span>
          Realtime
        </span>
        <span class="user-pill" v-if="authStore.user">
          <vue-feather type="user" size="14" />
          {{ authStore.user.displayName }}
        </span>
      </div>
    </div>

    <!-- ── Analytics: KPI Cards + Charts + Detection Log ─────────────── -->
    <AnalyticsCharts />

  </main>
</template>

<script setup>
import VueFeather      from 'vue-feather';
import AnalyticsCharts from '../components/AnalyticsCharts.vue';
import { useAuthStore } from '../stores/authStore';

const authStore = useAuthStore();
</script>

<style scoped>
.dashboard-page {
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 32px;
  padding-bottom: 40px;
}

.page-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 12px;
}

.page-title h1 {
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 1.6rem;
  font-weight: 800;
  color: #1a202c;
  letter-spacing: -0.03em;
  margin-bottom: 4px;
}
.title-icon { color: #667eea; }

.page-subtitle {
  color: #718096;
  font-size: 0.87rem;
  font-weight: 500;
  margin-left: 36px;
}

.header-actions { display: flex; align-items: center; gap: 10px; }

.live-pill {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 14px;
  border-radius: 20px;
  background: rgba(16,185,129,.08);
  border: 1px solid rgba(16,185,129,.25);
  font-size: 0.75rem;
  font-weight: 700;
  color: #059669;
}
.live-dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  background: #10b981;
  animation: pulse-dot 1.5s ease-in-out infinite;
}
@keyframes pulse-dot {
  0%, 100% { opacity: 1; transform: scale(1); }
  50%       { opacity: 0.5; transform: scale(1.4); }
}

.user-pill {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 14px;
  border-radius: 20px;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  font-size: 0.78rem;
  font-weight: 600;
  color: #4a5568;
}
</style>
