// frontend/src/router/index.ts
import { createRouter, createWebHistory } from 'vue-router';
import { useAuthStore } from '../stores/authStore';

import Login     from '../views/Login.vue';
import Dashboard from '../views/Dashboard.vue';
import Profile   from '../views/Profile.vue';

const routes = [
  {
    path: '/login',
    name: 'Login',
    component: Login,
    meta: { title: 'Đăng nhập' }
  },
  {
    path: '/',
    redirect: '/dashboard'
  },
  {
    path: '/dashboard',
    name: 'Dashboard',
    component: Dashboard,
    meta: { title: 'Dashboard — Highway Guardian', requiresAuth: true }
  },
  {
    path: '/profile',
    name: 'Profile',
    component: Profile,
    meta: { title: 'Hồ sơ — Highway Guardian', requiresAuth: true }
  }
];

const router = createRouter({
  history: createWebHistory(),
  routes
});

router.beforeEach((to, _from, next) => {
  const authStore    = useAuthStore();
  const requiresAuth = to.matched.some(r => r.meta.requiresAuth);

  if (requiresAuth && !authStore.isLoggedIn) {
    next({ name: 'Login' });
  } else if (to.name === 'Login' && authStore.isLoggedIn) {
    next({ name: 'Dashboard' });
  } else {
    next();
  }
});

export default router;