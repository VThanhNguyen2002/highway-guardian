// frontend/src/router/index.js
import { createRouter, createWebHistory } from 'vue-router';
// BỎ watch, watchOnce
import { useAuthStore } from '../stores/authStore';

// (Import các view...)
import Login from '../views/Login.vue';
import Detect from '../views/Detect.vue';
import Camera from '../views/Camera.vue';
import History from '../views/History.vue';
import Map from '../views/Map.vue';

const routes = [
  {
    path: '/login',
    name: 'Login',
    component: Login,
    meta: { title: 'Đăng nhập' }
  },
  {
    path: '/',
    name: 'Detect',
    component: Detect,
    meta: { title: 'Nhận diện (Tải ảnh)', requiresAuth: true }
  },
  {
    path: '/camera',
    name: 'Camera',
    component: Camera,
    meta: { title: 'Nhận diện (Camera)', requiresAuth: true }
  },
  {
    path: '/history',
    name: 'History',
    component: History,
    meta: { title: 'Lịch sử Phát hiện', requiresAuth: true }
  },
  {
    path: '/map',
    name: 'Map',
    component: Map,
    meta: { title: 'Bản đồ Biển báo', requiresAuth: true }
  }
];

const router = createRouter({
  history: createWebHistory(),
  routes
});

// SỬA LẠI TOÀN BỘ GUARD
router.beforeEach((to, _from, next) => {
  const authStore = useAuthStore();
  const requiresAuth = to.matched.some(record => record.meta.requiresAuth);

  // Vì main.js đã 'await checkAuth()',
  // nên khi code này chạy, authStore.isLoggedIn đã là CHÍNH XÁC
  
  if (requiresAuth && !authStore.isLoggedIn) {
    // Nếu trang yêu cầu login VÀ user chưa login -> Về Login
    next({ name: 'Login' });
  } else if (to.name === 'Login' && authStore.isLoggedIn) {
    // Nếu đã login mà vào trang login -> Về trang chủ
    next({ name: 'Detect' });
  } else {
    // Các trường hợp khác
    next();
  }
});

export default router;