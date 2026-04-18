<template>
  <div class="login-container">
    <div class="login-box">
      <vue-feather type="shield" class="logo-icon"></vue-feather>
      <h2>Highway Guardian</h2>
      <p>Chào mừng trở lại! Vui lòng đăng nhập.</p>
      
      <form @submit.prevent="handleLogin">
        
        <div class="input-group">
          <input 
            type="email" 
            placeholder="Tên đăng nhập (Email)" 
            v-model="email" 
            required
            :disabled="authStore.loading"
          >
          <vue-feather type="user" class="input-icon"></vue-feather>
        </div>
        
        <div class="input-group">
          <input 
            type="password" 
            placeholder="Mật khẩu" 
            v-model="password" 
            required
            :disabled="authStore.loading"
          >
          <vue-feather type="lock" class="input-icon"></vue-feather>
        </div>
        
        <p v-if="errorMessage" class="error-message">{{ errorMessage }}</p>

        <button type="submit" class="primary" :disabled="authStore.loading">
          <span v-if="!authStore.loading">Đăng nhập</span>
          <span v-else class="loading-text">
            <span class="spinner-small"></span>
            Đang xử lý...
          </span>
        </button>
      </form>
    </div>

    <!-- Toast Notification -->
    <Toast 
      :visible="toast.visible"
      :type="toast.type"
      :title="toast.title"
      :message="toast.message"
      :duration="toast.duration"
      @close="toast.visible = false"
    />
  </div>
</template>

<script setup>
import { ref, reactive } from 'vue';
import VueFeather from 'vue-feather';
import Toast from '../components/Toast.vue';
import { useAuthStore } from '../stores/authStore';
import { useRouter } from 'vue-router';

// Khởi tạo
const authStore = useAuthStore();
const router = useRouter();

// Biến cho form
const email = ref('');
const password = ref('');
const errorMessage = ref(null);

// Toast state
const toast = reactive({
  visible: false,
  type: 'success',
  title: '',
  message: '',
  duration: 3000
});

// Hàm hiển thị toast
const showToast = (type, title, message, duration = 3000) => {
  toast.type = type;
  toast.title = title;
  toast.message = message;
  toast.duration = duration;
  toast.visible = true;
};

// Hàm xử lý login
async function handleLogin() {
  // Reset thông báo
  errorMessage.value = null;

  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/; 
  
  // 1. Validation
  if (!email.value || !password.value) {
    errorMessage.value = "Vui lòng nhập đầy đủ email và mật khẩu.";
    return;
  }
  if (!emailRegex.test(email.value)) {
    errorMessage.value = "Định dạng email không hợp lệ.";
    return;
  }
  
  // Gọi action 'login' từ store
  const result = await authStore.login(email.value, password.value);

  // 2. Kiểm tra kết quả
  if (typeof result === 'boolean' && result === true) {
    showToast('success', 'Đăng nhập thành công!', 'Đang chuyển hướng...', 1500);
    // Redirect immediately — the router guard will take over
    router.push({ name: 'Dashboard' });

  } else {
    // Xử lý lỗi
    const errorCode = result;
    let message = 'Đã xảy ra lỗi. Vui lòng thử lại.';

    switch (errorCode) {
      case 'auth/user-not-found':
      case 'auth/wrong-password':
      case 'auth/invalid-credential':
        message = 'Email hoặc mật khẩu không chính xác.';
        break;
      case 'auth/invalid-email':
        message = 'Địa chỉ email không hợp lệ.';
        break;
      case 'auth/too-many-requests':
        message = 'Quá nhiều lần thử. Vui lòng thử lại sau.';
        break;
      case 'auth/user-doc-not-found':
        message = 'Không tìm thấy thông tin người dùng.';
        break;
    }
    
    // Hiển thị lỗi qua Toast
    showToast('error', 'Đăng nhập thất bại', message, 4000);
  }
}
</script>

<style scoped>
.login-container {
  width: 100%;
  height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  position: relative;
  overflow: hidden;
}

.login-container::before {
  content: '';
  position: absolute;
  width: 500px;
  height: 500px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 50%;
  top: -250px;
  right: -250px;
}

.login-container::after {
  content: '';
  position: absolute;
  width: 400px;
  height: 400px;
  background: rgba(255, 255, 255, 0.08);
  border-radius: 50%;
  bottom: -200px;
  left: -200px;
}

.login-box {
  width: 100%;
  max-width: 440px;
  padding: 48px;
  background: rgba(255, 255, 255, 0.98);
  backdrop-filter: blur(10px);
  border-radius: 20px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(255, 255, 255, 0.3);
  text-align: center;
  position: relative;
  z-index: 1;
  animation: slideUp 0.6s ease-out;
}

@keyframes slideUp {
  from {
    opacity: 0;
    transform: translateY(30px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.logo-icon {
  width: 64px;
  height: 64px;
  color: #667eea;
  margin-bottom: 16px;
  filter: drop-shadow(0 4px 8px rgba(102, 126, 234, 0.3));
  animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% {
    transform: scale(1);
  }
  50% {
    transform: scale(1.05);
  }
}

.login-box h2 {
  color: #1a202c;
  margin-bottom: 8px;
  font-size: 2rem;
  font-weight: 700;
  letter-spacing: -0.025em;
}

.login-box p {
  margin-bottom: 32px;
  color: #718096;
  font-size: 0.95rem;
  font-weight: 500;
}

.input-group {
  position: relative;
  margin-bottom: 20px;
}

.input-icon {
  position: absolute;
  left: 18px;
  top: 50%;
  transform: translateY(-50%);
  width: 20px;
  height: 20px;
  color: #a0aec0;
  transition: color 0.3s ease;
  z-index: 1;
}

.input-group input:focus + .input-icon {
  color: #667eea;
}

.input-group input {
  width: 100%;
  padding: 14px 18px 14px 52px;
  border: 2px solid #e2e8f0;
  border-radius: 10px;
  box-sizing: border-box;
  font-size: 0.95rem;
  font-weight: 500;
  color: #2d3748;
  transition: all 0.3s ease;
  background: #fff;
}

.input-group input:focus {
  border-color: #667eea;
  outline: none;
  box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
}

.input-group input::placeholder {
  color: #cbd5e0;
}

button.primary {
  width: 100%;
  margin-top: 8px;
  padding: 14px 18px;
  font-size: 1rem;
}

button.primary:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.loading-text {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
}

.spinner-small {
  width: 16px;
  height: 16px;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-top-color: #fff;
  border-radius: 50%;
  display: inline-block;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.error-message {
  color: #e53e3e;
  background: rgba(229, 62, 62, 0.1);
  border-left: 4px solid #e53e3e;
  padding: 12px 16px;
  border-radius: 8px;
  margin-bottom: 20px;
  text-align: left;
  font-size: 0.9rem;
  font-weight: 600;
  animation: shake 0.5s ease;
}

@keyframes shake {
  0%, 100% { transform: translateX(0); }
  25% { transform: translateX(-10px); }
  75% { transform: translateX(10px); }
}

</style>