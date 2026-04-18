<!-- frontend/src/components/Sidebar.vue -->
<template>
  <nav class="sidebar">
    <div class="sidebar-header">
      <vue-feather type="shield" class="logo-icon"></vue-feather>
      <h2>Highway Guardian</h2>
    </div>

    <ul class="nav-links">
      <li>
        <router-link to="/dashboard">
          <vue-feather type="bar-chart-2" class="nav-icon"></vue-feather>
          <span>Dashboard</span>
        </router-link>
      </li>
      <li>
        <router-link to="/dashboard#detection-logs">
          <vue-feather type="list" class="nav-icon"></vue-feather>
          <span>Detection Logs</span>
        </router-link>
      </li>
      <li>
        <router-link to="/profile">
          <vue-feather type="user" class="nav-icon"></vue-feather>
          <span>Hồ sơ</span>
        </router-link>
      </li>
    </ul>
      
    <!-- Phần logout tách riêng -->
    <div class="sidebar-footer">
       <a href="#" @click.prevent="handleLogout" class="logout-link">
          <vue-feather type="log-out" class="nav-icon"></vue-feather>
          <span>Đăng xuất</span>
        </a>
    </div>
  </nav>
</template>

<script setup>
import VueFeather from 'vue-feather';
import { useAuthStore } from '../stores/authStore'; // Đảm bảo đường dẫn đúng
import { useRouter } from 'vue-router';

const authStore = useAuthStore();
const router = useRouter();

const handleLogout = () => {
  authStore.logout();
  router.push({ name: 'Login' });
};
</script>

<style scoped>
.sidebar {
  width: 260px;
  background: linear-gradient(180deg, #1a202c 0%, #2d3748 100%);
  color: #e2e8f0;
  padding: 28px 16px;
  height: 100vh;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  box-shadow: 2px 0 10px rgba(0, 0, 0, 0.1);
  position: relative;
  overflow-y: auto;
}

.sidebar::-webkit-scrollbar {
  width: 6px;
}

.sidebar::-webkit-scrollbar-track {
  background: transparent;
}

.sidebar::-webkit-scrollbar-thumb {
  background: #4a5568;
  border-radius: 3px;
}

/* Header */
.sidebar-header {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  margin-bottom: 40px;
  padding-bottom: 24px;
  border-bottom: 2px solid rgba(102, 126, 234, 0.3);
}

.sidebar-header h2 {
  font-size: 1.5rem;
  font-weight: 700;
  margin: 0;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.logo-icon {
  width: 32px;
  height: 32px;
  color: #667eea;
  filter: drop-shadow(0 2px 4px rgba(102, 126, 234, 0.4));
}

/* Danh sách link chính */
.nav-links {
  list-style: none;
  padding: 0;
  margin: 0;
  flex-grow: 1;
}

.nav-links li a,
.logout-link {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 14px 18px;
  text-decoration: none;
  color: #cbd5e1;
  border-radius: 10px;
  margin-bottom: 6px;
  transition: all 0.3s ease;
  font-weight: 500;
  font-size: 0.95rem;
  position: relative;
  overflow: hidden;
}

.nav-links li a::before,
.logout-link::before {
  content: '';
  position: absolute;
  left: 0;
  top: 0;
  height: 100%;
  width: 3px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  transform: scaleY(0);
  transition: transform 0.3s ease;
}

/* Hiệu ứng hover */
.nav-links li a:hover,
.logout-link:hover {
  background: rgba(102, 126, 234, 0.15);
  color: #fff;
  transform: translateX(4px);
}

.nav-links li a:hover::before,
.logout-link:hover::before {
  transform: scaleY(1);
}

/* Link đang active */
.nav-links li a.router-link-active {
  background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
  color: #fff;
  font-weight: 600;
  box-shadow: 0 4px 6px rgba(102, 126, 234, 0.2);
}

.nav-links li a.router-link-active::before {
  transform: scaleY(1);
}

.nav-links li a.router-link-active .nav-icon {
  color: #667eea;
}

/* Icon trong link */
.nav-icon {
  width: 20px;
  height: 20px;
  flex-shrink: 0;
  color: #94a3b8;
  transition: all 0.3s ease;
}

.nav-links li a:hover .nav-icon,
.logout-link:hover .nav-icon {
  color: #667eea;
  transform: scale(1.1);
}

/* Footer (chứa logout) */
.sidebar-footer {
  margin-top: auto;
  padding-top: 20px;
  border-top: 2px solid rgba(102, 126, 234, 0.3);
}

.logout-link {
  margin-bottom: 0;
  color: #fc8181;
}

.logout-link:hover {
  background: rgba(252, 129, 129, 0.15);
  color: #feb2b2;
}

.logout-link:hover .nav-icon {
  color: #feb2b2;
}
</style>