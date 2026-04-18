<template>
  <div class="profile-page">
    <div class="profile-card">
      <!-- Avatar -->
      <div class="avatar-ring">
        <div class="avatar">
          <vue-feather type="user" size="40" />
        </div>
      </div>

      <!-- User info -->
      <div class="profile-info">
        <h1 class="profile-name">{{ authStore.user?.displayName || 'Người dùng' }}</h1>
        <p class="profile-email">
          <vue-feather type="mail" size="14" />
          {{ authStore.user?.email || '—' }}
        </p>
        <span class="role-badge">
          <vue-feather type="shield" size="12" />
          Admin
        </span>
      </div>

      <hr class="divider" />

      <!-- Details grid -->
      <ul class="detail-grid">
        <li>
          <span class="detail-label">
            <vue-feather type="hash" size="13" /> User ID
          </span>
          <span class="detail-value uid">{{ authStore.user?.uid || '—' }}</span>
        </li>
        <li>
          <span class="detail-label">
            <vue-feather type="clock" size="13" /> Phiên đăng nhập
          </span>
          <span class="detail-value">{{ sessionTime }}</span>
        </li>
      </ul>

      <!-- Logout -->
      <button class="logout-button" @click="handleLogout">
        <vue-feather type="log-out" size="16" />
        Đăng xuất
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import VueFeather from 'vue-feather';
import { useAuthStore } from '../stores/authStore';
import { useRouter } from 'vue-router';

const authStore = useAuthStore();
const router    = useRouter();

// Show a live "session active" timer
const sessionStart = ref(Date.now());
const now          = ref(Date.now());

const sessionTime = computed(() => {
  const diffMs = now.value - sessionStart.value;
  const mins   = Math.floor(diffMs / 60_000);
  const secs   = Math.floor((diffMs % 60_000) / 1000);
  return `${String(mins).padStart(2, '0')}m ${String(secs).padStart(2, '0')}s`;
});

let timer: ReturnType<typeof setTimeout>;
onMounted(() => {
  timer = setInterval(() => { now.value = Date.now(); }, 1000) as unknown as ReturnType<typeof setTimeout>;
});

async function handleLogout() {
  clearInterval(timer);
  await authStore.logout();
  router.push({ name: 'Login' });
}
</script>

<style scoped>
.profile-page {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  padding: 32px 16px;
  background: #f0f4ff;
}

.profile-card {
  background: #fff;
  border-radius: 20px;
  padding: 48px 40px;
  width: 440px;
  max-width: 100%;
  box-shadow: 0 12px 40px rgba(102, 126, 234, 0.12);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 20px;
  text-align: center;
}

/* Avatar */
.avatar-ring {
  width: 92px;
  height: 92px;
  border-radius: 50%;
  background: linear-gradient(135deg, #667eea, #764ba2);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 3px;
}

.avatar {
  width: 86px;
  height: 86px;
  border-radius: 50%;
  background: linear-gradient(135deg, #eef2ff, #ede9fe);
  display: flex;
  align-items: center;
  justify-content: center;
  color: #667eea;
}

/* Info */
.profile-info {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
}

.profile-name {
  margin: 0;
  font-size: 1.5rem;
  font-weight: 800;
  color: #1a202c;
  letter-spacing: -0.02em;
}

.profile-email {
  margin: 0;
  display: flex;
  align-items: center;
  gap: 6px;
  color: #64748b;
  font-size: 0.9rem;
}

.role-badge {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 3px 12px;
  background: linear-gradient(135deg, rgba(102,126,234,0.12), rgba(118,75,162,0.12));
  color: #667eea;
  font-size: 0.78rem;
  font-weight: 700;
  border-radius: 20px;
  letter-spacing: 0.03em;
}

.divider {
  width: 100%;
  border: none;
  border-top: 1.5px solid #f1f5f9;
  margin: 4px 0;
}

/* Details */
.detail-grid {
  list-style: none;
  padding: 0;
  margin: 0;
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.detail-grid li {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  text-align: left;
}

.detail-label {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.82rem;
  font-weight: 600;
  color: #94a3b8;
  white-space: nowrap;
  flex-shrink: 0;
}

.detail-value {
  font-size: 0.85rem;
  color: #334155;
  font-weight: 500;
  word-break: break-all;
  text-align: right;
}

.detail-value.uid {
  font-family: monospace;
  font-size: 0.75rem;
  color: #64748b;
}

/* Logout button */
.logout-button {
  margin-top: 8px;
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  padding: 13px;
  border: none;
  border-radius: 12px;
  font-size: 0.95rem;
  font-weight: 700;
  cursor: pointer;
  background: linear-gradient(135deg, #ff6b6b, #ee0979);
  color: #fff;
  transition: all 0.2s;
  letter-spacing: 0.02em;
}

.logout-button:hover {
  opacity: 0.9;
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(238, 9, 121, 0.3);
}

.logout-button:active {
  transform: translateY(0);
}
</style>
