<template>
  <header class="header">
    <div class="header-content">
      <h1 class="header-title">{{ $route.meta.title || 'Dashboard' }}</h1>
      
      <div class="user-profile" v-if="authStore.isLoggedIn && authStore.user">
        <span>Xin chào, {{ authStore.user.displayName }}!</span> 
      </div>
      
    </div>
  </header>
</template>

<script>
// Import store
import { useAuthStore } from '../stores/authStore';

export default {
  name: 'Header',
  // Dùng setup() để khởi tạo store
  setup() {
    const authStore = useAuthStore();
    
    // Trả về store để template có thể dùng
    return { authStore };
  }
}
</script>

<style scoped>
.header {
  height: 70px;
  flex-shrink: 0;
  background: #ffffff;
  border-bottom: 1px solid #e2e8f0;
  padding: 0 32px;
  display: flex;
  align-items: center;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
  position: relative;
  z-index: 10;
}

.header-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
}

.header-title {
  font-size: 1.75rem;
  color: #1a202c;
  font-weight: 700;
  margin: 0;
  letter-spacing: -0.025em;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.user-profile {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 8px 16px;
  background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
  border-radius: 20px;
  color: #2d3748;
  font-weight: 600;
  font-size: 0.95rem;
  transition: all 0.3s ease;
}

.user-profile:hover {
  background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
  transform: translateY(-1px);
  box-shadow: 0 4px 6px rgba(102, 126, 234, 0.2);
}

.avatar {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  object-fit: cover;
  border: 2px solid #667eea;
}
</style>